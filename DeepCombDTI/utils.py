# import numpy and pandas
import numpy as np
import pandas as pd

# import keras modules
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2,l1
from keras.preprocessing import sequence
from sklearn.metrics import precision_recall_curve, auc, roc_curve

# import other modules
import os

# parse data
def parse_data(dti_dir, drug_dir, protein_dir, drug_vecs, drug_lens, prot_vec, prot_len):
    print('Parsing {0}, {1}'.format(*[dti_dir ,drug_dir]))

    # set column names
    drug_col = "Compound_ID"
    protein_col = "Protein_ID"
    label_col = "Label"
        
    # load DTI, drug, protein data
    dti_df = pd.read_csv(dti_dir)
    drug_df = pd.read_csv(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")

    # Extract drug features
    drug_features = []
    for drug_vec in drug_vecs:
        drug_dic = drug_df[drug_vec].map(lambda fp: fp.split("\t")).to_dict()
        drug_feature = np.array(list(dti_df[drug_col].map(lambda drug: drug_dic[drug])), dtype=np.float64)
        drug_features.append(drug_feature)

    # Extract protein features
    prot_dic= protein_df[prot_vec].map(lambda seq: [float(i) for i in seq.split("\t")]).to_dict()
    protein_feature = np.array(list(dti_df[protein_col].map(lambda protein: prot_dic[protein])), dtype=np.float64)
    
    # Extract labels
    label =  dti_df[label_col].values

    print("\tPositive data : %d" %(sum(dti_df[label_col])))
    print("\tNegative data : %d" %(dti_df.shape[0] - sum(dti_df[label_col])))
    return {"drug_features": drug_features, "protein_feature": protein_feature, "label": label}

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser(description="""
        This Python script is used to train, validate, test deep learning model for prediction of drug-target interaction (DTI)\n
        Deep learning model will be built by Keras with tensorflow.\n
        You can set almost hyper-parameters as you want, See below parameter description\n
        DTI, drug and protein data must be written as csv file format. And feature should be tab-delimited format for script to parse data.\n
        \n
        requirement\n
        ============================\n
        tensorflow > 1.0\n
        keras > 2.0\n
        numpy\n
        pandas\n
        scikit-learn\n
        ============================\n
        \n
        contact : dlsrnsladlek@gist.ac.kr\n
        """
    )
    
    # train_params
    parser.add_argument("dti_dir", help="Training DTI information [drug, target, label]")
    parser.add_argument("drug_dir", help="Training drug information [drug, SMILES,[feature_name, ..]]")
    parser.add_argument("protein_dir", help="Training protein information [protein, seq, [feature_name]]")
    
    # test_params
    parser.add_argument("--test-name", '-n', help="Name of test data sets", nargs="*")
    parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]", nargs="*")
    parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]", nargs="*")
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", nargs="*")
    parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI")
    
    # structure_params (drug)
    parser.add_argument("--drug-vecs", "-V", help="Types of drug feature", nargs="*", type=str, default="")
    parser.add_argument("--drug-lens", "-L", help="Drug vector lengths", default=None, nargs="*", type=int)
    parser.add_argument("--drug-layers-list", '-c', help="Dense layers for drugs", default=None, type=str)
    parser.add_argument("--drug-layer-methods", "-M", help="Methods for drug layers", default=None, nargs="*", type=str)
    parser.add_argument("--fc-drug-layers","-fd", help="Dense layers for concatenated drug layers", default=None, nargs="*", type=int)
    
    # structure_params (protein)
    parser.add_argument("--prot-vec", "-v", help="Type of protein feature", type=str, default="")
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
    parser.add_argument("--protein-layers","-p", help="Dense layers for protein", default=None, nargs="*", type=int)
    
    # structure_params (fully connected)
    parser.add_argument("--fc-layers", '-f', help="Dense layers for concatenated layers of drug and target layer", default=None, nargs="*", type=int)
    
    # training_params
    parser.add_argument("--learning-rate", '-r', help="Learning late for training", default=1e-4, type=float)
    parser.add_argument("--n-epoch", '-e', help="The number of epochs for training or validation", type=int, default=10)
    
    # the other hyper-parameters
    parser.add_argument("--activation", "-a", help='Activation function of model', type=str)
    parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.2, type=float)
    parser.add_argument("--batch-size", "-b", help="Batch size", default=32, type=int)
    parser.add_argument("--decay", "-y", help="Learning rate decay", default=0.0, type=float)
    
    # mode_params
    parser.add_argument("--validation", help="Excute validation with independent data, will give AUC and AUPR (No prediction result)", action="store_true")
    parser.add_argument("--predict", help="Predict interactions of independent test set", action="store_true")
    parser.add_argument("--has-previous-model", "-hpm", help="Use previous trained models", action="store_true")
    
    # output_params
    parser.add_argument("--save-model", "-m", help="save model", type=str)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)
    
    # gpu_num_param
    parser.add_argument("--gpus", "-G", help="Select the number of activated GPUs", default=1, type=int)
    parser.add_argument("--gpu-num", "-g", help="Select GPU number", default="0", type=str)
    
    return parser.parse_args()

def print_parameter_summary(target, params):
    print('=====================================================')
    print(target, 'summary')
    print('=====================================================')
    for key, value in params.items():
        print('{:20s} : {:10s}'.format(key, str(value)))
    print('=====================================================')

def get_params(args):
    # set train data directories
    train_dic = {
        "dti_dir": args.dti_dir,
        "drug_dir": args.drug_dir,
        "protein_dir": args.protein_dir,
    }
    
    # training parameter
    train_params = {
        "n_epoch": args.n_epoch,
        "batch_size": args.batch_size,
    }
    
    # type parameter
    type_params = {
        "drug_vecs": args.drug_vecs,
        "drug_lens": args.drug_lens,
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
    }
    
    # model parameter
    model_params = {
        'drug_layers_list': args.drug_layers_list.split(','),
        'drug_layer_methods': args.drug_layer_methods,
        'fc_drug_layers': args.fc_drug_layers,
        'protein_layers': args.protein_layers,
        'fc_layers': args.fc_layers,
        'learning_rate': args.learning_rate,
        'decay': args.decay,
        'activation': args.activation,
        'dropout': args.dropout,
        'gpus': args.gpus,
        'gpu_num': args.gpu_num,
        'has_previous_model': args.has_previous_model
    }
    model_params.update(type_params)
    
    # get a ouput file name
    output_file = args.output

    # print a summary of model parameters
    print_parameter_summary('Model parameters', model_params)
    
    # set train data
    train_dic.update(type_params)
    train_dic = parse_data(**train_dic)
    
    # set test data
    test_sets = zip(args.test_name, args.test_dti_dir, args.test_drug_dir, args.test_protein_dir)
    test_dic = {
        test_name: parse_data(test_dti, test_drug, test_protein, **type_params)
        for test_name, test_dti, test_drug, test_protein in test_sets
    }
    
    
    return train_dic, test_dic, train_params, type_params, model_params, output_file

def select_gpu_num(gpu_num):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    print('GPU:', gpu_num, 'is used.')

def run_validation(dti_prediction_model, train_params, output_file, train_dic, test_dic):
    print('Validation')
    
    # extract and print validation parameters
    validation_params = {}
    validation_params.update(train_params)
    validation_params["output_file"] = output_file
    print_parameter_summary('Validation parameters', validation_params)
    validation_params.update(train_dic)
    validation_params.update(test_dic)
    
    # validate the model
    dti_prediction_model.validation(**validation_params)

def run_prediction(dti_prediction_model, train_params, output_file, train_dic, test_dic):
    print('Prediction')
    
    # fit the model and predict
    train_dic.update(train_params)
    dti_prediction_model.fit(**train_dic)
    test_predicted = dti_prediction_model.predict(**test_dic)
    
    # save prediction results as dataframe
    result_df = pd.DataFrame()
    result_columns = []
    for dataset in test_predicted:
        # extract prediction results
        value = np.squeeze(test_predicted[dataset]['predicted'])
        print(dataset+str(value.shape))
        
        # save prediction results as dataframe
        temp_df = pd.DataFrame()
        temp_df[dataset,'predicted'] = value
        temp_df[dataset, 'label'] = np.squeeze(test_predicted[dataset]['label'])
        result_df = pd.concat([result_df, temp_df], ignore_index=True, axis=1)
        result_columns.append((dataset, 'predicted'))
        result_columns.append((dataset, 'label'))
        
    # save prediction results to a csv file
    print('save to %s' % output_file)
    result_df.columns = pd.MultiIndex.from_tuples(result_columns)
    result_df.to_csv(output_file, index=False)