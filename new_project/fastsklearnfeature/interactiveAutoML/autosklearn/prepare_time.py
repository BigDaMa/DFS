import openml

my_dict = {31: {'construction_time': 37.22868013381958, 'hpo_time': 71.55725455284119, 'test_auc': 0.808046683046683},179: {'construction_time': 12.547806978225708, 'hpo_time': 26.018290519714355, 'test_auc': 0.8968253968253966}, 1464: {'construction_time': 2.119044542312622, 'hpo_time': 3.8105359077453613, 'test_auc': 0.7914598540145986}, 37: {'construction_time': 9.284051895141602, 'hpo_time': 25.370245456695557, 'test_auc': 0.866030399434429}, 50: {'construction_time': 5.624495506286621, 'hpo_time': 13.474086284637451, 'test_auc': 0.9991650853889943}, 334: {'construction_time': 2.942595958709717, 'hpo_time': 5.059381723403931, 'test_auc': 0.8849110032362459}, 3: {'construction_time': 50.17716026306152, 'hpo_time': 112.5660994052887, 'test_auc': 0.9993131868131868}, 1480: {'construction_time': 12.026309490203857, 'hpo_time': 28.70481824874878, 'test_auc': 0.7608941070219011}, 15: {'construction_time': 6.130415678024292, 'hpo_time': 13.487451553344727, 'test_auc': 0.9977698483496877}}


for k,v in my_dict.items():

	dataset = openml.datasets.get_dataset(k).name

	print(dataset + ',' + str(v['construction_time']) + ',' + str(v['hpo_time']))

