'''
Created on Aug 23, 2016

@author: mjchao
'''
import IPython
import model
import train_config


def main():
    config = train_config.DefaultConfig()
    char_pred_model = model.CharacterPredictorModel(config)
    char_pred_model.Train()
    _start_shell(locals())

def _start_shell(local_ns=None):
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


if __name__ == "__main__": main()
