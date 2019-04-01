# import numpy and pandas
import numpy as np
import pandas as pd

# import tensorflow, keras modules
import tensorflow as tf
import keras.backend as K
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
from DeepCombDTI.utils import select_gpu_num


class Drug_Target_Prediction(object):
    def modelv(
        self, drug_vecs, drug_lens, drug_layers_list, drug_layer_methods, fc_drug_layers,
        prot_vec, prot_len, protein_layers, fc_layers,
        activation, dropout, has_previous_model, initializer
    ):
        # return a tuple which have drug layer size and dimensions (e.g. (128, 64, 32, 16))
        def return_tuple(value):
            if type(value) is int:
               return [value]
            else:
               return tuple(value)

        # Set hyperparameters
        regularizer_param = 0.001
        params_dic = {"kernel_initializer": initializer, "kernel_regularizer": l2(regularizer_param)}
        
        if not has_previous_model:
            # construct drug layers
            input_ds = []; model_ds = []
            for drug_len, drug_layers, method in zip(drug_lens, drug_layers_list, drug_layer_methods):
                # create a input layer
                input_d = Input(shape=(drug_len, ))

                # create hidden layers
                if method == 'CNN' or method == 'GRU':
                    input_layer_d = Reshape((drug_len, 1,))(input_d)
                else:
                    input_layer_d = input_d
                prev_layer_size = drug_len
                drug_layers = return_tuple([int(layer) for layer in drug_layers.split(' ')])
                for layer_size in drug_layers:
                    # Initialize a model by the method
                    if method == 'CNN':
                        model_d = Convolution1D(
                            filters=10,
                            kernel_size=50,
                            strides=2,
                            padding='valid',
                            kernel_initializer=params_dic['kernel_initializer'],
                            kernel_regularizer=params_dic['kernel_regularizer']
                        )(input_layer_d)
                        model_d = GlobalMaxPooling1D(data_format='channels_first')(model_d)
                    elif method == 'GRU':
                        model_d = GRU(
                            layer_size,
                            input_shape=(1, prev_layer_size),
                            return_sequences=True
                        )(input_layer_d)
                        model_d = GlobalMaxPooling1D()(model_d)
                        prev_layer_size = layer_size
                    else:
                        model_d = input_layer_d
                        
                    model_d = Dense(layer_size, **params_dic)(model_d)
                    model_d = BatchNormalization()(model_d)
                    model_d = Activation(activation)(model_d)
                    model_d = Dropout(dropout)(model_d)
                    if method == 'CNN' or 'GRU':
                        input_layer_d = Reshape((layer_size, 1, ))(model_d)
                    else:
                        input_layer_d = model_d

                # add input and drug layers
                input_ds.append(input_d)
                model_ds.append(model_d)

            # construct fully connected drug layers
            model_d = Concatenate(axis=1)(model_ds)
            fc_drug_layers = return_tuple(fc_drug_layers)
            for fc_drug_layer in fc_drug_layers:
                model_d = Dense(units=fc_drug_layer, **params_dic)(model_d)
                model_d = BatchNormalization()(model_d)
                model_d = Activation(activation)(model_d)

            # construct protein layers (input layer, hidden layers)
            input_p = Input(shape=(prot_len,))
            input_layer_p = input_p
            protein_layers = return_tuple(protein_layers)
            for protein_layer in protein_layers:
                model_p = Dense(protein_layer, **params_dic)(input_layer_p)
                model_p = BatchNormalization()(model_p)
                model_p = Activation(activation)(model_p)
                model_p = Dropout(dropout)(model_p)
                input_layer_p = model_p
        else:
            MODEL_DIR = '/home/share/hwkim/model/' + fp_name + '/'
            m_filename = os.listdir(MODEL_DIR)[0]
            

        # construct fully connected layers
        # model_t = Concatenate(axis=1)([model_d, model_p])
        model_t = Concatenate(axis=1)(model_ds + [model_p])
        fc_layers = return_tuple(fc_layers)
        for fc_layer in fc_layers:
            model_t = Dense(units=fc_layer, **params_dic)(model_t)
            model_t = BatchNormalization()(model_t)
            model_t = Activation(activation)(model_t)
                
        # construct a prediction layer
        model_t = Dense(1, activation='tanh', activity_regularizer=l2(regularizer_param), **params_dic)(model_t)
        model_t = Lambda(lambda x: (x+1.)/2.)(model_t)

        # construct a model
        model_f = Model(inputs=input_ds + [input_p], outputs=model_t)

        return model_f

    def __init__(self, drug_vecs, drug_lens, drug_layers_list, drug_layer_methods, fc_drug_layers,
                 prot_vec, prot_len, protein_layers, fc_layers,
                 learning_rate, decay, activation, dropout, gpus, gpu_num, has_previous_model):
        # set hyperparameters
        self.__drug_vecs = drug_vecs
        self.__drug_lens = drug_lens
        self.__drug_layers_list = drug_layers_list
        self.__drug_layer_methods = drug_layer_methods
        self.__fc_drug_layers = fc_drug_layers
        self.__prot_vec = prot_vec
        self.__prot_len = prot_len
        self.__protein_layers = protein_layers
        self.__fc_layers = fc_layers
        self.__learning_rate = learning_rate
        self.__decay = decay
        self.__activation = activation
        self.__dropout = dropout
        self.__has_previous_model = has_previous_model
        self.__model_t = self.modelv(
            drug_vecs=self.__drug_vecs, drug_lens=self.__drug_lens, drug_layers_list=self.__drug_layers_list,
            drug_layer_methods=self.__drug_layer_methods, fc_drug_layers=self.__fc_drug_layers,
            prot_vec=self.__prot_vec, prot_len=self.__prot_len, protein_layers=self.__protein_layers,
            fc_layers=self.__fc_layers, activation=self.__activation, dropout=self.__dropout, initializer='glorot_normal',
            has_previous_model=self.__has_previous_model
        )
        if gpus >= 2:
            self.__model_t = multi_gpu_model(self.__model_t, gpus=gpus)
        else:
            select_gpu_num(gpu_num)
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        set_session(tf.Session(config=config))

        # create a optimizer and a model
        opt = Adam(lr=learning_rate, decay=self.__decay)
        self.__model_t.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        K.get_session().run(tf.global_variables_initializer())

    def fit(self, drug_features, protein_feature, label, n_epoch, batch_size):
        for _ in range(n_epoch):
            history = self.__model_t.fit(
                drug_features.append(protein_feature), label,
                initial_epoch=_, epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1
            )
        return self.__model_t
    
    def summary(self):
        self.__model_t.summary()
    
    def validation(self, drug_features, protein_feature, label, output_file=None, n_epoch=10, batch_size=32, **kwargs):
        # create result dataframe and dictionary
        if output_file:
            param_tuple = pd.MultiIndex.from_tuples([("parameter", param) for param in ['drug_layers', 'fc_layers', 'learning_rate']])
            result_df = pd.DataFrame(
                data=[[','.join(self.__drug_layers_list), self.__fc_layers, self.__learning_rate]]*n_epoch,
                columns=param_tuple
            )
            result_df['epoch'] = range(1,n_epoch+1)
        result_dic = {dataset: {'AUC': [], 'AUPR': [], 'opt_threshold(AUPR)':[], 'opt_threshold(AUC)':[] }for dataset in kwargs}
        
        for _ in range(n_epoch):
            history = self.__model_t.fit(
                drug_features + [protein_feature], label,
                initial_epoch=_, epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1
            )
            for dataset in kwargs:
                print('\tPredction of', dataset)
                
                # prediction
                test_ds = kwargs[dataset]['drug_features']
                test_p = kwargs[dataset]['protein_feature']
                test_label = kwargs[dataset]['label']
                prediction = self.__model_t.predict(test_ds + [test_p])
                
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

    def predict(self, **kwargs):
        results_dic = {}
        for dataset in kwargs:
            # get test data
            result_dic = {}
            test_ds = kwargs[dataset]['drug_features']
            test_p = kwargs[dataset]['protein_feature']
            result_dic['label'] = kwargs[dataset]['label']
            
            # predict and save prediction results
            result_dic['predicted'] = self.__model_t.predict(test_ds + [test_p])
            results_dic[dataset] = result_dic
            
        return results_dic
    
    def save(self, output_file):
        self.__model_t.save(output_file)
