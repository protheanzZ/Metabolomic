library(xcms)

load('ms2data.RData')
start = Sys.time()
print(paste("start get Spectra of features on positive mode", start))

ms2_p <- featureSpectra(ms2data_p)
print(paste("Done.. release swap if there is. Time spent: ", Sys.time()-start))

system('echo 123qwe | sudo -S swapoff -a')
system('echo 123qwe | sudo -S swapon -a')

print(paste("start get Spectra of features on negative mode", start))
ms2_n <- featureSpectra(ms2data_n)
print(paste("Done.. release swap if there is. Time spent: ", Sys.time()-start))

system('echo 123qwe | sudo -S swapoff -a')
system('echo 123qwe | sudo -S swapon -a')

print('Save data..')
save(ms2_p,ms2_n, file='MS2spectra.R')

print("ALL FINISHED!!!")

m = 1
n = NA
p = 0
for (i in c(1:length(ms2_n))){
    if (ms2_n[[i]]@peaksCount > m){
        m = ms2_n[[i]]@peaksCount
        p = i
    }
}
m
n
get_ms2_df <- function(ms2obj){
    df = data.frame()
    for (i in c(1:length(ms2obj))){
        print(i)
        feature_id = ms2obj[i]@elementMetadata[1,1]
        peak_id = ms2obj[i]@elementMetadata[1,2]
        precursormz = ms2obj[[i]]@precursorMz
        mz = ms2obj[[i]]@mz
        int = ms2obj[[i]]@intensity
        temp <- data.frame()
        temp[1,1] <- feature_id
        colnames(temp)<- 'feature_id'
        temp$precursor_mz<-precursormz
        temp$peak_id <- peak_id
        temp$mz <- list(mz)
        temp$intensity <- list(int)
        df <- rbind(df, temp)
    }
    df
}
ms2pdf <- get_ms2_df(ms2_p)
ms2ndf <- get_ms2_df(ms2_n)

ms2pdf$mz <- vapply(ms2pdf$mz, paste, collapse = ", ", character(1L))
ms2pdf$intensity <- vapply(ms2pdf$intensity, paste, collapse = ", ", character(1L))

ms2ndf$mz <- vapply(ms2ndf$mz, paste, collapse = ", ", character(1L))
ms2ndf$intensity <- vapply(ms2ndf$intensity, paste, collapse = ", ", character(1L))

write.csv(ms2pdf, file='ms2p.csv')
write.csv(ms2ndf, file='ms2n.csv')
p <- chromPeaks(ms2data_n)