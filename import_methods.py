from methods.herding import *
from methods.kmeans import *
# from methods.cdal import *
from methods.kcentergreedy import *
from methods.lowdimention import *
# from methods.UNAC import *
# from methods.cal import *
# from methods.cal_k import *
from methods.dbal import *
from methods.svp_max_entropy import *
from methods.mcal import *
from methods.dfal import *
from methods.boundary_aware import *
from methods.glister import *
from methods.cg_influence import *
from methods.LiSSA import *
from methods.Arnoldi import *
from methods.TracIn import *
from methods.TMCS import *
from methods.KNN_shapley import *
from methods.cs_shapley import *
import methods.dvrl as dvrl
# from methods.craig_base import *
from methods.craig import *
from methods.gradmatch import *
from methods.crest import *
from methods.full import *
from methods.IAS import *
from methods.random import *
from methods.G_shapley import *


def run_method(args,channel,im_size,num_classes,class_names,mean,std, dst_train,dst_val,dst_test):
    if args.method == 'kmeans':
        kmeans_selection(args,channel,im_size,num_classes,class_names,mean,std, dst_train,dst_val,dst_test)
    elif args.method == 'herding':
        herding_selection(args,channel,im_size,num_classes,class_names,mean,std, dst_train,dst_val,dst_test)
    elif args.method == 'full':
        full_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'kcentergreedy':
        kcentergreedy_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'lowdimention':
        lowdimention_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    # elif args.method == 'UNAC':
    #     UNAC_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    # elif args.method == 'cal':
    #     cal_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    # elif args.method == 'calk':
    #     cal_k_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'dbal':
        dbal_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'svp_EP':
        svp_max_entropy_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'mcal':
        mcal_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'dfal':
        adversarial_deepfool_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'boundary_aware':
        boundary_aware_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'glister':
        glister_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'cg_influence':
        cg_influence_selection(args,channel,im_size,num_classes,class_names,mean,std, dst_train,dst_val,dst_test)
    elif args.method == 'LiSSA':
        LiSSA_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'Arnoldi':
        Arnoldi_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'TracIn':
        TracIn_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'TMCS':
        TMCS_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'KNN_shapley':
        KNN_Shapley_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'cs_shapley':
        CS_Shapley_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'dvrl':
        dvrl.DVRL_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    # elif args.method == 'craig_base':
    #     craig_base_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'craig':
        craig_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'gradmatch':
        gradmatch_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'crest':
        crest_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    # elif args.method == 'retrieve':
    #     retrieve_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'IAS':
        IAS_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'random':    
        random_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)
    elif args.method == 'G_shapley':
        G_shapley_selection(args,channel,im_size,num_classes,class_names,mean,std,dst_train,dst_val,dst_test)