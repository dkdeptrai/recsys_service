def binary(list, referenceList):
    binaryList = []
    for i in referenceList:
        if i in list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList

def get_index(df, shoe_id):
    return df[df['asin'] == shoe_id].index[0]
