library(rhdf5)

info <- h5ls(args[1])
keys <- unique(info[,1])
keys = keys[2:length(keys)]

h5createFile("archie_64_predictions.hdf5")

load("trained_model_ArchIE_64.Rdata")

for (key in keys[1:100]) {
  print(key)
  data = aperm(h5read(args[1], paste(key, "/features", sep = "")))
  dim_ = dim(data)
  
  # instantiate an array for the results
  X <- array(rep(0, dim_[1]*dim_[2]*dim_[3]), c(dim_[1], dim_[2], dim_[3]))
  
  for (i in 1:dim_[1]) {
    for (j in 1:dim_[2]) {
      X[i, j, 1:dim_[3]] <- plogis(predict.glm(model_cleaned, as.data.frame(data[i,j,1:dim_[3],1:73])))
    }
  }
  h5createGroup("archie_64_predictions.hdf5", key)
  h5createDataset("archie_64_predictions.hdf5", paste(key, "/features", sep = ""), dim(X))
  h5write(X, file="archie_64_predictions.hdf5",
          name=paste(key, "/features", sep = ""))
  
}

h5closeAll()

