# import numpy and pandas
import numpy as np
import pandas as pd

# import tensorflow, keras modules
import tensorflow as tf
import keras.backend as K
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding, Lambda
from keras.layers import Convolution1D, Reshape, GlobalMaxPooling1D, SpatialDropout1D, GRU
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2,l1
from keras.preprocessing import sequence
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from keras.utils.training_utils import multi_gpu_model
from keras.backend.tensorflow_backend import set_session
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
import os


class Drug_Target_Prediction(object):
    def model(self, drug_vecs, drug_lens, drug_layers_list,
              prot_vec, prot_len, protein_layers_list, fc_layers_list,
              activation, dropout, initializer):
        # return a tuple which have drug layer size and dimensions (e.g. (128, 64, 32, 16))
        def return_tuple(value):
            if type(value) is int:
               return [value]
            else:
               return tuple(value)

        # Set hyperparameters
        regularizer_param = 0.001
        params_dic = {"kernel_initializer": initializer, "kernel_regularizer": l2(regularizer_param)}
        
        inputs = []; model_ts = []
        for drug_len, drug_layers, protein_layers, fc_layers in zip(drug_lens, drug_layers_list, protein_layers_list, fc_layers_list):
            # construct drug layers
            input_d = Input(shape=(drug_len, ))
            inputs.append(input_d)
            input_layer_d = input_d
            for layer_size in drug_layers:
                model_d = input_layer_d
                model_d = Dense(layer_size, **params_dic)(model_d)
                model_d = BatchNormalization()(model_d)
                model_d = Activation(activation)(model_d)
                model_d = Dropout(dropout)(model_d)
                input_layer_d = model_d

            # construct protein layers (input layer, hidden layers)
            input_p = Input(shape=(prot_len,))
            inputs.append(input_p)
            input_layer_p = input_p
            protein_layers = return_tuple(protein_layers)
            for protein_layer in protein_layers:
                model_p = Dense(protein_layer, **params_dic)(input_layer_p)
                model_p = BatchNormalization()(model_p)
                model_p = Activation(activation)(model_p)
                model_p = Dropout(dropout)(model_p)
                input_layer_p = model_p

            # construct fully connected layers
            model_t = Concatenate(axis=1)([model_d, model_p])
            for fc_layer in fc_layers:
                model_t = Dense(units=fc_layer, **params_dic)(model_t)
                model_t = BatchNormalization()(model_t)
                model_t = Activation(activation)(model_t)
                    
            # construct a prediction layer
            model_t = Dense(1, activation='tanh', activity_regularizer=l2(regularizer_param), **params_dic)(model_t)
            model_t = Lambda(lambda x: (x+1.)/2.)(model_t)
            model_ts.append(model_t)
        
         
        # construct a ensemble model
        max_model_t = layers.average(model_ts) 
        model_ens = Model(inputs=inputs, outputs=max_model_t, name='ensemble')  

        # optimize a model
        opt = Adam(lr=self.__learning_rate, decay=self.__decay)
        model_ens.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model_ens

    def __init__(self, drug_vecs, drug_lens, drug_layers_list,
                 prot_vec, prot_len, protein_layers_list, fc_layers_list,
                 learning_rate, decay, activation, dropout, model_output):
        # set hyperparameters
        self.__drug_vecs = drug_vecs
        self.__drug_lens = drug_lens
        self.__drug_layers_list = drug_layers_list
        self.__prot_vec = prot_vec
        self.__prot_len = prot_len
        self.__protein_layers_list = protein_layers_list
        self.__fc_layers_list = fc_layers_list
        self.__learning_rate = learning_rate
        self.__decay = decay
        self.__activation = activation
        self.__dropout = dropout
        self.__model_output = model_output
        self.__model_t = self.model(drug_vecs=self.__drug_vecs, drug_lens=self.__drug_lens, drug_layers_list=self.__drug_layers_list,
                                    prot_vec=self.__prot_vec, prot_len=self.__prot_len, protein_layers_list=self.__protein_layers_list,
                                    fc_layers_list=self.__fc_layers_list, activation=self.__activation, dropout=self.__dropout, initializer='glorot_normal')

        config = tf.ConfigProto()
        set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())

    def fit(self, features, label, n_epoch, batch_size):
        for epoch in range(n_epoch):
            if not os.path.exists(self.__model_output + f'/{epoch}'): os.makedirs(self.__model_output + f'/{epoch}')
            model_path = self.__model_output + f'_{epoch+1:03d}.ckpt'
            if os.path.exists(model_path): self.__model_t.load_weights(model_path)
            checkpoint = ModelCheckpoint(filepath=model_path, save_weights_only=True, verbose=0)
            self.__model_t.fit(features, label, initial_epoch=epoch, epochs=epoch+1, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[checkpoint])

        return self.__model_t
    
    def summary(self):
        return self.__model_t.summary()
    
    def validation(self, features, label, output_file=None, n_epoch=10, batch_size=32, **kwargs):
        # create result dataframe and dictionary
        if output_file:
            param_tuple = pd.MultiIndex.from_tuples([("parameter", param) for param in ['drug_layers', 'fc_layers', 'learning_rate']])
            result_df = pd.DataFrame(
                data=[[','.join(self.__drug_layers_list), self.__fc_layers, self.__learning_rate]]*n_epoch,
                columns=param_tuple
            )
            result_df['epoch'] = range(1,n_epoch+1)
        result_dic = {dataset: {'AUC': [], 'AUPR': [], 'opt_threshold(AUPR)':[], 'opt_threshold(AUC)':[] }for dataset in kwargs}
        
        for epoch in range(n_epoch):
            if not os.path.exists(self.__model_output + f'/{epoch}'): os.makedirs(self.__model_output + f'/{epoch}')
            model_path = self.__model_output + f'_{epoch+1:03d}.ckpt'
            if os.path.exists(model_path): self.__model_t.load_weights(model_path)
            checkpoint = ModelCheckpoint(filepath=model_path, save_weights_only=True, verbose=0)
            self.__model_t.fit(features, label, initial_epoch=epoch, epochs=epoch+1, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[checkpoint])
            for dataset in kwargs:
                print('\tPredction of', dataset)
                
                # prediction
                test_f = kwargs[dataset]['features']
                test_label = kwargs[dataset]['label']
                prediction = self.__model_t.predict(test_f)
                
                # get performances (AUC, AUPR)
                fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
                AUC = auc(fpr, tpr)
                precision, recall, thresholds = precision_recall_curve(test_label, prediction)
                AUPR = auc(recall,precision)
                
                # get optimal thresholds (AUC, AUPR)
                distance = (1-fpr)**2 + (1-tpr)**2
                EERs = (1-recall) / (1-precision)
                positive = sum(test_label)
                negative = test_label.shape[0] - positive
                ratio = negative / positive
                opt_t_AUC = thresholds_AUC[np.argmin(distance)]
                opt_t_AUPR = thresholds[np.argmin(np.abs(EERs-ratio))]
                
                # print performances
                print('\tArea Under ROC Curve(AUC): %0.3f' % AUC)
                print('\tArea Under PR Curve(AUPR): %0.3f' % AUPR)
                print('\tOptimal threshold(AUC)   : %0.3f' % opt_t_AUC)
                print('\tOptimal threshold(AUPR)  : %0.3f' % opt_t_AUPR)
                print('=================================================')
                
                # save prediction results as a dictionary
                result_dic[dataset]['AUC'].append(AUC)
                result_dic[dataset]['AUPR'].append(AUPR)
                result_dic[dataset]['opt_threshold(AUC)'].append(opt_t_AUC)
                result_dic[dataset]['opt_threshold(AUPR)'].append(opt_t_AUPR)
                
        # save prediction results as a csv file
        if output_file:
            for dataset in kwargs:
                result_df[dataset, 'AUC'] = result_dic[dataset]['AUC']
                result_df[dataset, 'AUPR'] = result_dic[dataset]['AUPR']
                result_df[dataset, 'opt_threshold(AUC)'] = result_dic[dataset]['opt_threshold(AUC)']
                result_df[dataset, 'opt_threshold(AUPR)'] = result_dic[dataset]['opt_threshold(AUPR)']
            print('save to', output_file)
            print(result_df)
            result_df.to_csv(output_file, index=False)

        return 

    def predict(self, **kwargs):
        results_dic = {}
        for dataset in kwargs:
            # get test data
            result_dic = {}
            test_f = kwargs[dataset]['features']
            test_label = kwargs[dataset]['label']
            
            # predict and save prediction results
            prediction = self.__model_t.predict(test_f)
            fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
            AUC = auc(fpr, tpr)
            precision, recall, thresholds = precision_recall_curve(test_label, prediction)
            AUPR = auc(recall,precision)

            # get optimal thresholds (AUC, AUPR)
            distance = (1-fpr)**2 + (1-tpr)**2
            EERs = (1-recall) / (1-precision)
            positive = sum(test_label)
            negative = test_label.shape[0] - positive
            ratio = negative / positive
            
            # print performances
            print('\tPredction of', dataset)
            print('\tArea Under ROC Curve(AUC): %0.3f' % AUC)
            print('\tArea Under PR Curve(AUPR): %0.3f' % AUPR)
            print('=================================================')

            result_dic['label'] = test_label
            result_dic['predicted'] = prediction
            results_dic[dataset] = result_dic
            
        return results_dic
    
    def save(self):
        self.__model_t.save(self.__model_output)
