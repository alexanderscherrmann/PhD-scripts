library(nat)
path="/atmosdyn2/ascherrmann/010-IFS/traj/MED/use/"
setwd(path)
columns <- c("time", "lon", "lat", "P", "PS", "PV", "ZB","OL", "PVRCONVT", "PVRCONVM", "PVRTURBT", "PVRTURBM", "PVRSW","PVRLWH", "PVRLWC", "PVRLS")

lis <- list.files(pattern="trajectories-mature")
ids <- c()

for (j in seq_along(lis)){
  f <- lis[j]
  d <- read.table(f)
  n <- length(d[, 1])
  id <- substring(f,36,41)

  df <- data.frame("time" = d[, 1])
  for (i in 2:length(columns)){
    assign(paste0(columns[i],id), matrix(d[, i], nrow  = n / 49, byrow = TRUE))
    assign(paste0("dipv",paste0(columns[i],id)),flip(t(apply(flip(get(paste0(columns[i],id)),flipdim="Y"),2,cumsum)),flipdim="Y")[,1])
  }
  
  #assign(paste0("pvrtot",id), PVRCONVT + PVRCONVM + PVRLS + PVRLWC + PVRSW + PVRLWH + PVRTURBM + PVRTURBT)
}



# av<-apply(m,2,mean)
# apv<-apply(PV,2,mean)