import io
import pickle
import zlib
import os

def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir
    
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def experiment_already_done(args):
    RESULTS_ARGS = results_dir(args)
    MODEL = os.path.join(RESULTS_ARGS, 'model.pth')
    if os.path.exists(MODEL):
        return True
    else:
        return False

def file2dict(DATASET):
    with open(DATASET, 'rb') as fd:
        zbytes = fd.read()
    # decompress bytes
    bytes = zlib.decompress(zbytes)
    return pickle.loads(bytes)
	
def read_pkl_acc(RESULTS_ARGS):
    DICT = os.path.join(RESULTS_ARGS, 'accuracy.pkl')
    acc_dict = load_dict(DICT)
    acc_test = acc_dict['test']
    acc_train = acc_dict['train']
    return acc_test, acc_train

def read_pkl_perf(RESULTS_ARGS):
    DICT = os.path.join(RESULTS_ARGS, 'performance.pkl')
    dict = load_dict(DICT)
    return dict
    
def experiment_already_done(args):
    RESULTS_ARGS = results_dir(args)
    MODEL = os.path.join(RESULTS_ARGS, 'model.pth')
    if os.path.exists(MODEL):
        return True
    else:
        return False
	
def results_dir(args):
    RESULTS_ARGS = os.path.join(args.RESULTS, 
                                '{0}_{1}_{2}_split={3}_fnt={4}_high={5}_epochs={6}_bs={7}_sched={8}_lrmax={9}_seed(data)={10}'.format(
                                args.dataset,
                                args.model,
                                args.method,
								args.split,
                                args.faint,
                                args.high,
                                args.num_epochs,
                                args.batch_size,
                                args.use_sched,
                                args.lr_max,
                                args.seed_data,
                                ))
    return RESULTS_ARGS