import argparse

def H_parse():
    parse=argparse.ArgumentParser("define the argument.")
    parse.add_argument("--train_epoch",default=1000,type=int,
                       help="train model epoch.")

    parse.add_argument("--batch_size",default=64,type=int,
                       help="train model data batch size.")

    parse.add_argument("--lr_rate",default=5e-4,type=float,
                       help="train model leran rate.")

    parse.add_argument("--data_dict",default="/",type=str,
                       help="data dict.")

    parse.add_argument("--data_name",default="filename",type=str,
                       help="data file name.")

    parse.add_argument("--epoch_st",default=0,type=int,
                       help="train model start epoch.")

    parse.add_argument("--model_path",default="",type=str,
                       help="where to keep model data.")

    parse.add_argument("--pre_train",action="store_true",
                       default=False,
                       help="Are you model pre_trained ?")

    parse.add_argument("--model_name",default="model_name",type=str,
                       help="model name.")

    return parse.parse_args()