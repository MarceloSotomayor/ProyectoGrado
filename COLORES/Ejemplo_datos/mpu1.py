from Utilitarios import MPU6050
import time
import datetime
i=0
muestreo=0.25
print "INICIANDO"
while(i<100):
    i=i+1
    imu = MPU6050()
    f=open('mpu_log.txt','a')
    now=datetime.datetime.now()
    timestamp = now.strftime("%Y/%m/%d %H:%M") 
    (acc_x, acc_y, acc_z) = imu.get_acc()
    (gyro_x, gyro_y, gyro_z) = imu.get_gyro()
    outstring=str(timestamp)+","+str(acc_x)+","+str(acc_y)+","+str(acc_z)+","+str(gyro_x)+","+str(gyro_y)+","+str(gyro_z)+"\n"
    f.write(outstring)
    f.close()
    time.sleep(muestreo)
print "FINALIZANDO"    
    
