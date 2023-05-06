
import torch
z=torch.zeros(2,3)
z2=torch.zeros(2,3)
z=z==1
#shift = torch.nn.functional.interpolate(z, size=(64,64) , mode='bilinear')
z1=~z
print(z)
print(z1*z2)

#print(shift.size())


# t1=True
# t2=False
#
# if not t2 and t1 :
#     print("1")
# else:
#     print("2")

