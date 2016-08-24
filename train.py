'''
Created on Aug 23, 2016

@author: mjchao
'''
import model
import train_config


def main():
    config = train_config.DefaultConfig()
    char_pred_model = model.CharacterPredictorModel(config)
    char_pred_model.Train()


if __name__ == "__main__": main()
