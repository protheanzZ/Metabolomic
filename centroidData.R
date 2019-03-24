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
