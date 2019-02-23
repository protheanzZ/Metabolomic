library(xcms)
library(RColorBrewer)
library(faahKO)
library(magrittr)
library(pander)

register(SerialParam())

fnames <- list.files(paste0(getwd(), '/data_centroid'), pattern='.*.mzXML$')

for (i in c(1:length(fnames))){
    fnames[i] <- paste0('data_centroid/', fnames[i])
}

fnames

raw_data_centroid <- readMSData(files=fnames, mode='onDisk')

cwp <- CentWaveParam(peakwidth = c(5,20), noise = 5000, ppm=5)
xdata_centroid <- findChromPeaks(raw_data_centroid, param = cwp)

xdata_centroid <- adjustRtime(xdata_centroid, param = ObiwarpParam(binSize = 0.6)) 

pdp <- PeakDensityParam(sampleGroups = rep(1, length(fileNames(xdata_centroid))),
                        minFraction = 0.3, bw=10)

xdata_centroid <- groupChromPeaks(xdata_centroid, param = pdp)

xdata_centroid <- fillChromPeaks(xdata_centroid)

head(featureValues(xdata_centroid, value = "into"))
