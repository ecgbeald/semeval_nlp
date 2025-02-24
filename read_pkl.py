import pickle

with open('/vol/bitbucket/rm521/cw/predictions_22_02-133205.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(data)