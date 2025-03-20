import pickle

with open('/home/robotics/Desktop/ISRO_IRoC/weights.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)