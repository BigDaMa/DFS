import openml


flows = openml.flows.list_flows()


feature_selection_classes = ['SelectPercentile', 'SelectKBest', 'GenericUnivariateSelect', 'SelectFpr', ' SelectFdr', 'SelectFwe', 'RFE', 'SelectFromModel', 'VarianceThreshold']
fs_in_pipeline = {}


pipeline_flows = []
select_kbest_flows = 0
for f in flows.values():
    if f['full_name'].startswith('sklearn.pipeline.Pipeline'):
        print(str(f['id']) + ': ' + str(f['full_name']))
        pipeline_flows.append(f['id'])

        #my_flow = openml.flows.get_flow(flow_id=f['id'])
        #print(my_flow)

        for fs_class in feature_selection_classes:
            if fs_class in f['full_name']:
                if not fs_class in fs_in_pipeline:
                    fs_in_pipeline[fs_class] = []
                fs_in_pipeline[fs_class].append(f['id'])


print("number of flows that use sklearn pipelines: " + str(len(pipeline_flows)))
count_runs = 0
ci = 0
for v in pipeline_flows:
    runs = openml.runs.list_runs(flow=[v], display_errors=False)
    count_runs += len(runs)
    print(str(ci) + ': ' + str(count_runs))
    ci += 1
print('all pipeline' + " runs: " + str(count_runs))

'''
for key, value in fs_in_pipeline.items():
    print("flows that contain feature selection: "  + str(key) + " => "+ str(len(value)))
    count_runs = 0
    for v in value:
        runs = openml.runs.list_runs(flow=[v], display_errors=False)
        count_runs += len(runs)
        print(count_runs)
    print(str(key) + " runs: " + str(count_runs))
'''

