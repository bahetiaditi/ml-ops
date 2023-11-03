from utils import get_all_h_param_comb,read_digits,split_train_dev_test

def test_for_hparam_combinations_count():
    gamma_list=[0.001,0.01,0.1,1]
    C_list=[1,10,100,1000]
    h_params={}
    h_params['gamma']=gamma_list
    h_params['C']=C_list
    h_params_combinations = get_all_h_param_comb(gamma_list,C_list)

    assert len(h_params_combinations) == len(gamma_list)*len(C_list)

def test_for_hparam_combinations_values():
    gamma_list=[0.001,0.01]
    C_list=[1]
    h_params={}
    h_params['gamma']=gamma_list
    h_params['C']=C_list
    h_params_combinations = get_all_h_param_comb(gamma_list,C_list)

    expected_param_combo_1 = {'gamma':0.001,'C':1}
    expected_param_combo_2 = {'gamma':0.01,'C':1}

    # Convert the list of tuples to a list of dictionaries
    h_params_combinations_dict = [{'gamma': gamma, 'C': C} for gamma, C in h_params_combinations]

    assert expected_param_combo_1 in h_params_combinations_dict
    assert expected_param_combo_2 in h_params_combinations_dict




def test_data_splitting():
    X,y = read_digits()
    X=X[:100,:,:]
    y=y[:100]

    test_size = .1
    dev_size=.6
    train_size=1 - test_size - dev_size

    X_train ,X_test,X_dev , y_train , y_test , y_dev = split_train_dev_test(X,y,test_size,dev_size)


    assert len(X_train) == 30, f"Expected X_train length to be 30, but got {len(X_train)}"
    assert len(X_test) == 10, f"Expected X_test length to be 10, but got {len(X_test)}"
    assert len(X_dev) == 60, f"Expected X_dev length to be 60, but got {len(X_dev)}"