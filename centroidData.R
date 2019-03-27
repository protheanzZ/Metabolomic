library(xcms)
library(RColorBrewer)
library(faahKO)
library(magrittr)
library(pander)
library(CAMERA)


register(MulticoreParam(2))

fnames <- list.files('../movingdisk/data_centroid/', pattern='.*.mzXML$')

for (i in c(1:length(fnames))){
    fnames[i] <- paste0('../movingdisk/data_centroid/', fnames[i])
}

fnames

raw_data_centroid <- readMSData(files=fnames, mode='onDisk', msLevel. = 1)

cwp <- CentWaveParam(peakwidth = c(5,20), noise = 5000, ppm=5)
xdata_centroid <- findChromPeaks(raw_data_centroid, param = cwp, BPPARAM = MulticoreParam(8))
register(MulticoreParam(2))

xdata_centroid <- adjustRtime(xdata_centroid, param = ObiwarpParam(binSize = 0.6)) 

register(MulticoreParam(8))
pdp <- PeakDensityParam(sampleGroups = rep(1, length(fileNames(xdata_centroid))),
                        minFraction = 0.3, bw=10)

xdata_centroid <- groupChromPeaks(xdata_centroid, param = pdp)

xdata_centroid <- fillChromPeaks(xdata_centroid)

head(featureValues(xdata_centroid, value = "into"))

xsg <- as(xdata_centroid, 'xcmsSet')

xsa <- xsAnnotate(xsg)
#Group after RT value of the xcms grouped peak
xsaF <- groupFWHM(xsa, perfwhm=0.6)
#Verify grouping
xsaC <- groupCorr(xsaF)
#Annotate isotopes, could be done before groupCorr
xsaFI <- findIsotopes(xsaC)
#Annotate adducts
xsaFA <- findAdducts(xsaFI, polarity="positive")
#Get final peaktable and store on harddrive
write.csv(getPeaklist(xsaFA),file="result_CAMERA.csv")

for (i in (1:length(fnames))){
    file <- filterFile(xdata_centroid, file=i, keepAdjustedRtime=TRUE)
    p <- paste0(file@phenoData@data[1]$sampleNames)
    p <- substr(p, 1, nchar(p)-6)
    
    write.csv(rtime(file), paste0('RtMzInt_RAW/', p, '.csv'))
    
    mzs <- mz(file)
    for (i in (1:length(mzs))){
        write.table(mzs[[i]], paste0('RtMzInt_RAW/', p, '_mz.txt'), append=TRUE)
    }
    remove(mzs)
    int <- intensity(file)
    for (i in (1:length(int))){
        write.table(int[[i]], paste0('RtMzInt_RAW/', p, '_int.txt'), append=TRUE)
    }
    remove(int)
}
library(parallel)
cores <- detectCores()
ExportRtMzInt <- function(i){
    file <- xcms::filterFile(xdata_centroid, file=i, keepAdjustedRtime=TRUE)
    p <- paste0(file@phenoData@data[1]$sampleNames)
    p <- substr(p, 1, nchar(p)-6)
    
    write.csv(xcms::rtime(file), paste0('RtMzInt_RAW/', p, '_rt.csv'))
    
    mzs <- xcms::mz(file)
    for (i in (1:length(mzs))){
        write.table(mzs[[i]], paste0('RtMzInt_RAW/', p, '_mz.txt'), append=TRUE)
    }
    remove(mzs)
    int <- xcms::intensity(file)
    for (i in (1:length(int))){
        write.table(int[[i]], paste0('RtMzInt_RAW/', p, '_int.txt'), append=TRUE)
    }
    remove(int)
    return(0)
}
cl <- makeCluster(8)
clusterExport(cl=cl, varlist=c("xdata_centroid"))
re <- parLapply(cl, 1:length(fnames), ExportRtMzInt)
stopCluster(cl)

length(mzs)
for (i in (1:length(mzs))){
    write.table(mzs[i], 'mz.txt', append=TRUE)
}
filters <- filterFile(xdata_centroid, file=c(1), keepAdjustedRtime=TRUE)
write.csv(rtime(filters), paste0())
filters <- filterFile(xdata_centroid, file=c(1:6))
chr_filters <- chromatogram(filters, aggregationFun='max', rt=c(280, 380))
plot(chr_filters, col=paste0(brewer.pal(6, "Set1")[1:6], "60"))