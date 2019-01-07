from cartman import solenoid

sn = solenoid()

for i in range(10):
    sn.drive(2,50)
    sn.drive(3,50)
    sn.drive(4,50)
    sn.drive(5,50)
    sn.drive(13,50)
