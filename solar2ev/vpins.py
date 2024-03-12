
###########
# prediction tab
###########

########## prediction 

# label (text) persistent when android app or device exit
# date, confidence
v_pred_label = 4 

# prediction bin. 
# v_binx should be x will be used as index 
v_bin0 = 0 # low production
v_bin1 = 1
v_bin2 = 2
v_bin3 = 3 # high production

########## post mortem

# label persistent when android app or device exit
# date, real, pred, confidence
v_post_mortem_label = 13 

# post mortem led
# red, green, blue
v_yesterday = 5
v_moins2 = 6
v_moins3 = 7
v_moins4 = 8
v_moins5 = 9
v_moins6 = 10


########## program charger
# drop down menu auto, program charge, do not program charge
v_charge_menu = 11 

#  12 not used

# real terminal or
# use labeled value, as persistent "text". when android app or device exit
v_terminal = 14 

###########
# charger tab
###########

# slider. hours of charger overnite
# 0 is left and lower production
v_charge_0 = 15
v_charge_1 = 16
v_charge_2 = 17
v_charge_3 = 18

v_label_charge = 19 

# slider. set confidence threshold
v_confidence = 21

###########
# model tab
###########

# current model. trained on x days
v_model_info = 20

# gauge
v_model_acc = 22
v_model_pre = 23
v_model_rec = 24

# label for test on unseen data %, nb samples, confidence
v_unseen = 25

# gauge
v_model_rmse = 26
v_model_mse = 27
v_model_mae = 28

# used by template header. very concise
v_header_template = 29

# used by app header. whatever is usefull 
v_header_app = 30

# aggregate of all postmortem (% sucess)
v_post_mortem_so_far = 31
