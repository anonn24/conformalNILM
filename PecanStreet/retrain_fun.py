import tensorflow_probability.substrates.jax as tfp
import flax.linen as nn
import jax
import jax.numpy as jnp
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime
from dataloader import *
from model import *
from utilities import *
from plotting import *
import datetime
# jax.config.update("jax_default_dtype_bits", '16')

dict_bs_lr = {'air': (2048, 0.0001), 'refrigerator': (4096, 0.01), 'furnace':(2048, 0.0001), 'dishwasher':(4096,0.0001), 'clotheswasher':(2048,0.0001)}


def retrain(new_house, new_start,new_end, pool_houses, train, test, given_date, appliance, data_aggregated):
    # train = data_aggregated[data_aggregated["dataid"].isin(train_houses)] 
    # test = data_aggregated[data_aggregated["dataid"].isin(test_houses)]
    n=99
    if new_house==0:
        x_train, y_train = dataloader(appliance, train,"2018-03-01 00:00:00-06" , "2018-03-10 23:59:00-06",n)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_train = scaler_x.fit_transform(x_train)
        y_train = scaler_y.fit_transform(y_train)
        x_train = jnp.array(x_train).reshape(x_train.shape[0], n, 1)
        y_train = jnp.array(y_train)
        model = seq2point()
        params =  model.init(jax.random.PRNGKey(0), x_train, True)
        params, losses = fit(model, params, x_train, y_train,False, batch_size=dict_bs_lr[appliance][0], learning_rate=dict_bs_lr[appliance][1], epochs=30)
        plt.plot(losses)
        sns.despine()
        plt.show()
        x_test, y_test = dataloader(appliance, test, "2018-05-01 00:00:00-06", "2018-05-10 23:59:00-06",n)
        x_test = scaler_x.transform(x_test)
        x_test = jnp.array(x_test).reshape(x_test.shape[0], n, 1)
        y_test = jnp.array(y_test)
        y_hat = model.apply(params, x_test, True, rngs={"dropout":jax.random.PRNGKey(0)})
        n_stacks = 10
        test_mean = scaler_y.inverse_transform(y_hat[0])
        test_sigma = scaler_y.scale_*y_hat[1]

        fn = lambda x, i : model.apply(params, x, False, rngs={"dropout": jax.random.PRNGKey(i)})
        y_stacks = jax.vmap(jax.jit(fn), in_axes=(None, 0))(x_test, jnp.arange(n_stacks))
        y_true = scaler_y.inverse_transform(y_test.reshape(-1,1))
        print(y_stacks[1].shape)
        cf_test_mean,cf_test_sigma = gmm_mean_var(y_stacks[0],y_stacks[1])
        cf_test_mean = scaler_y.inverse_transform(cf_test_mean)
        cf_test_sigma = (scaler_y.scale_*cf_test_sigma)
        print(f"RMSE : {rmse(y_test, test_mean):.4f} MAE  : {mae(y_test,test_mean):.4f} NLL : {NLL(test_mean,test_sigma,y_test):.4f}")
        plot_predict(y_test,test_mean,cf_test_sigma)

        # fig, ax = plt.subplots(1)
        # df, df3 = calibration_regression(cf_test_mean.squeeze(), cf_test_sigma.squeeze(), 
        #                                     y_test.squeeze(), "test", "b", ax)

        # _ = ax.set_title(f'Test = {ace(df3):.4f}')
        # ax.legend()

    else:
        new_df = data_aggregated[((data_aggregated['dataid']==new_house)&(data_aggregated['localminute']>new_start))]
        train = train.append(new_df)
        print("Train houses are")
        print(train["dataid"].unique())
        x_test, y_test = dataloader(appliance, test, "2018-05-01 00:00:00-06", "2018-05-10 23:59:00-06",n)

        end_date=new_end
        x_train, y_train = dataloader(appliance, train, "2018-03-01 00:00:00-06", end_date,n)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_train = scaler_x.fit_transform(x_train)
        y_train = scaler_y.fit_transform(y_train)
        x_train = jnp.array(x_train).reshape(x_train.shape[0], n, 1)
        y_train = jnp.array(y_train)
        model = seq2point()
        params =  model.init(jax.random.PRNGKey(0), x_train, True)
        params, losses = fit(model, params, x_train, y_train,False, batch_size=dict_bs_lr[appliance][0], learning_rate=dict_bs_lr[appliance][1], epochs=30)
        plt.plot(losses)
        sns.despine()
        plt.show()
        x_test = scaler_x.transform(x_test)
        x_test = jnp.array(x_test).reshape(x_test.shape[0], n, 1)
        y_test = jnp.array(y_test)
        n_stacks = 10
        y_hat = model.apply(params, x_test, True, rngs={"dropout":jax.random.PRNGKey(0)})
        test_mean = scaler_y.inverse_transform(y_hat[0])
        test_sigma = scaler_y.scale_*y_hat[1]

        fn = lambda x, i : model.apply(params, x, False, rngs={"dropout": jax.random.PRNGKey(i)})
        y_stacks = jax.vmap(jax.jit(fn), in_axes=(None, 0))(x_test, jnp.arange(n_stacks))
        y_true = scaler_y.inverse_transform(y_test.reshape(-1,1))
        cf_test_mean,cf_test_sigma = gmm_mean_var(y_stacks[0],y_stacks[1])
        cf_test_mean = scaler_y.inverse_transform(cf_test_mean)
        cf_test_sigma = (scaler_y.scale_*cf_test_sigma)

        print(f"RMSE : {rmse(y_test, test_mean):.4f} MAE  : {mae(y_test,test_mean):.4f} NLL : {NLL(test_mean,test_sigma,y_test):.4f}")
        plot_predict(y_test,test_mean,cf_test_sigma)

        # fig, ax = plt.subplots(1)
        # df, df3 = calibration_regression(cf_test_mean.squeeze(), cf_test_sigma.squeeze(), 
        #                                     y_test.squeeze(), "test", "b", ax)

        # _ = ax.set_title(f'Test = {ace(df3):.4f}')
        # ax.legend()


    max_uncertainity = 0
    max_house_id = 0
    max_house = 0
    

    for i in range(len(pool_houses)):
        pool_data = data_aggregated[data_aggregated['dataid']==pool_houses[i]]
        date_str = given_date+ '00'# Add a day and fix the format
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S%z")
        date_plus_1_day = date + datetime.timedelta(days=1)
        date_end = str(date_plus_1_day)
        mc_test_sigma_list = []
        for j in range(15):
            x_pool, y_pool = dataloader(appliance, pool_data, date_str, date_end, n)

            x_pool = scaler_x.transform(x_pool)
            x_pool = np.array(x_pool).reshape(x_pool.shape[0], n, 1)

            n_stacks = 10
            fn = lambda x, i : model.apply(params, x, False, rngs={"dropout": jax.random.PRNGKey(i)})
            y_stacks = jax.vmap(jax.jit(fn), in_axes=(None, 0))(x_pool, jnp.arange(n_stacks))
            # y_true = scaler_y.inverse_transform(y_test.reshape(-1,1))
            mc_test_mean,mc_test_sigma = gmm_mean_var(y_stacks[0],y_stacks[1])
            mc_test_mean = scaler_y.inverse_transform(mc_test_mean)
            mc_test_sigma = (scaler_y.scale_*mc_test_sigma)
            mc_test_sigma_list.append(mc_test_sigma.mean())

        weights = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125])
        mc_test_sigma_weighted = np.array(mc_test_sigma_list)*weights

        test_sigma_mean = mc_test_sigma_weighted.mean()
        # test_mean = scaler_y.inverse_transform(y_hat[0])
        if(test_sigma_mean>max_uncertainity):
            max_uncertainity = test_sigma_mean
            max_house_id = i
            max_house = pool_houses[i]
    return max_house, max_house_id, rmse(y_test, test_mean), train, test


def retrain_random(new_house, new_start,new_end, pool_houses, train, test,appliance, data_aggregated):
    
    current_y_hat= -1
    n=99
    if new_house==0:
        # getting the train data

        # print(f"new house is {new_house}\n new-start is {new_start}\n new_end is {new_end}\n pool_houses are {pool_houses}\ntrain is {train}\ntest is {test}\n batch_size is {batch_size}\n learning_rate is {learning_rate}")
        # print(f"Shape of train is {train.shape} and shape of test is {test.shape}")
        x_train, y_train = dataloader(appliance, train,"2018-03-01 00:00:00-06" , "2018-03-10 23:59:00-06",n)
        
        # initializing standard scalers
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        # transforming the data
        x_train = scaler_x.fit_transform(x_train)
        y_train = scaler_y.fit_transform(y_train)
        x_train = jnp.array(x_train).reshape(x_train.shape[0], n, 1)
        y_train = jnp.array(y_train)
        
        # initialziing the model
        model = seq2point()
        #getting the model parameters
        params =  model.init(jax.random.PRNGKey(0), x_train, True)
        params, losses = fit(model, params, x_train, y_train,False, batch_size=dict_bs_lr[f'{appliance}'][0], learning_rate=dict_bs_lr[f'{appliance}'][1], epochs=30)
        #plotting the losses
        plt.plot(losses)
        sns.despine()
        #getting the test data
        x_test, y_test = dataloader(appliance, test, "2018-05-01 00:00:00-06", "2018-05-10 23:59:00-06",n)
        #scaling the test data
        x_test = scaler_x.transform(x_test)
        #covnerting to jnp array
        x_test = jnp.array(x_test).reshape(x_test.shape[0], n, 1)
        y_test = jnp.array(y_test)
        #getting results
        y_hat = model.apply(params, x_test, True, rngs={"dropout":jax.random.PRNGKey(0)})
        
        test_mean = scaler_y.inverse_transform(y_hat[0])
        test_sigma = scaler_y.scale_*y_hat[1]

        print(f"RMSE : {rmse(y_test, test_mean):.4f} MAE  : {mae(y_test,test_mean):.4f} NLL : {NLL(test_mean,test_sigma,y_test):.4f}")
        
        current_y_hat = test_mean

    else:

        #fetching new house's dataframe, in the required time frame
        new_df = data_aggregated[((data_aggregated['dataid']==new_house)&(data_aggregated['localminute']>new_start))]
        
        #adding to train data
        train = train.append(new_df)
        print("Train houses are")
        print(train["dataid"].unique())

        #getting the test data
        x_test, y_test = dataloader(appliance, test, "2018-05-01 00:00:00-06", "2018-05-10 23:59:00-06",n)
        #setting the new end date
        end_date=new_end

        #getting the train data
        x_train, y_train = dataloader(appliance, train, "2018-03-01 00:00:00-06", end_date,n)
        print(x_train.shape)

        #initializing the scaler
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        #scaling the data
        x_train = scaler_x.fit_transform(x_train)
        y_train = scaler_y.fit_transform(y_train)
        #converting to jnp array
        x_train = jnp.array(x_train).reshape(x_train.shape[0], n, 1)
        y_train = jnp.array(y_train)

        #initializing the model
        model = seq2point()

        #getting model params
        params =  model.init(jax.random.PRNGKey(0), x_train, True)
        params, losses = fit(model, params, x_train, y_train,False, batch_size=dict_bs_lr[f'{appliance}'][0], learning_rate=dict_bs_lr[f'{appliance}'][1], epochs=30)
        # plt.plot(losses)
        sns.despine()

        # transforming the test data
        x_test = scaler_x.transform(x_test)
        x_test = jnp.array(x_test).reshape(x_test.shape[0], n, 1)
        y_test = jnp.array(y_test)
        n_stacks = 10

        #getting the predictions
        y_hat = model.apply(params, x_test, True, rngs={"dropout":jax.random.PRNGKey(0)})
        #getting test mean and sigma
        test_mean = scaler_y.inverse_transform(y_hat[0])
        test_sigma = scaler_y.scale_*y_hat[1]
        current_y_hat = test_mean
        print(f"RMSE : {rmse(y_test, test_mean):.4f} MAE  : {mae(y_test,test_mean):.4f} NLL : {NLL(test_mean,test_sigma,y_test):.4f}")
    
    return rmse(y_test, test_mean), mae(y_test,test_mean), train, test, current_y_hat
        
        