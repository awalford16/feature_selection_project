from sklearn.feature_selection import RFE

def recursive_feature_selection(model, k):
    return RFE(model, k)