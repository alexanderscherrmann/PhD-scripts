f <- read.table("/atmosdyn2/ascherrmann/010-IFS/traj/MED/use/trajectories-mature-20181020_08-ID-000102.txt")

columns <- c("time", "lon", "lat", "P", "PS", "PV", "ZB","OL", "PVRCONVT", "PVRCONVM", "PVRTURBT", "PVRTURBM", "PVRSW","PVRLWH", "PVRLWC", "PVRLS")

n <- length(f[, 1])

df <- data.frame("time" = f[, 1])
for (i in 2:length(columns)){
  df[,i] <- f[, i]
  colnames(df)[i] <- columns[i]
  assign(columns[i], matrix(df[, i], nrow  = n / 49, byrow = TRUE))
}
assign("pvrtot", PVRCONVT + PVRCONVM + PVRLS + PVRLWC + PVRSW + PVRLWH + PVRTURBM + PVRTURBT)

m<-round(matrix(runif(nrow(pvrtot)*ncol(pvrtot)),nrow(pvrtot),ncol(pvrtot)))
av<-apply(m,2,mean)
apv<-apply(PV,2,mean)
pv10<-apply(PV,2,quantile,probs=c(0.1))
pv90<-apply(PV,2,quantile,probs=c(0.9))

png(file="/home/ascherrmann/scripts/Paper1/R-test.png",width=600,height=400)
matplot(f[1:49,1],cbind(apv,pv10,pv90),type="l",lty=1,
         col=c("black","#9e1313","#21b1b1"))
polygon(c(f[1:49,1],rev(f[1:49,1])),c(pv10,rev(pv90)),col="#53535999")
#plot(f[1:49,1],apv,color="black")
#line(f[1:49,1],pv10,color="#9e1313")
#line(f[1:49,1],pv90,color="#21b1b1")
dev.off()