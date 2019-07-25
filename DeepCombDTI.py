import DeepCombDTI
from DeepCombDTI.DeepCombDTI import Drug_Target_Prediction
from DeepCombDTI.utils import get_args, get_params, run_validation, run_prediction

if __name__ == '__main__':
    
    # get parameters from arguments
    args = get_args()
    train_dic, test_dic, train_params, type_params, model_params, output_file = get_params(args)

    # construct DTI prediction model
    dti_prediction_model = Drug_Target_Prediction(**model_params)
    # print(dti_prediction_model.summary())

    # run validation or prediction
    if args.validation:
        run_validation(dti_prediction_model, train_params, output_file, train_dic, test_dic)
    elif args.predict:
        run_prediction(dti_prediction_model, train_params, output_file, train_dic, test_dic)

    # save trained model
    if args.save_model:
        dti_prediction_model.save()
    exit()
