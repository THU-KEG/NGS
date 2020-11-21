from __future__ import print_function

def is_NA(x):
    return 0 in x
def f_score(pred, labels, subtype_pred, subtype_golden):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    predict = []
    golden = []
    for i in range(len(pred)):
        predict.append((subtype_pred[i], pred[i]))
        golden.append((subtype_golden[i], labels[i]))
    for i in range(len(predict)):
        if predict[i]==golden[i] and not is_NA(predict[i]):
            TP+=1
        elif predict[i]!=golden[i]:
            if is_NA(predict[i]) and not is_NA(golden[i]):
                FN+=1
            elif is_NA(golden[i]) and not is_NA(predict[i]):
                FP+=1
            elif (not is_NA(golden[i])) and (not is_NA(predict[i])):
                FN+=1
                FP+=1
            else:
                TN+=1
        else:
            TN+=1
    try:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F = 2*P*R/(P+R)
    except:
        P=R=F=0
    return P,R,F