library(xcms)
library(RColorBrewer)
library(faahKO)
library(magrittr)
library(pander)

register(SnowParam(1))

filenames <- list.files(paste0(getwd(), '/data'), pattern='.*.mzML$')

for (i in c(1:length(filenames))){
    filenames[i] <- paste0('data/', filenames[i])
}

filenames

raw_data <- readMSData(files=filenames, mode='onDisk')

cwp <- CentWaveParam(peakwidth = c(20, 80), noise = 5000, ppm=5)
xdata <- findChromPeaks(raw_data, param = cwp)

xdata_adjustRT <- adjustRtime(xdata, param = ObiwarpParam(binSize = 0.6)) 

pdp <- PeakDensityParam(sampleGroups = rep(1, length(fileNames(xdata))),
                        minFraction = 0.35, bw = 20)

xdata_adjustRT <- groupChromPeaks(xdata_adjustRT, param = pdp)

xdata_adjustRT <- fillChromPeaks(xdata_adjustRT)

head(featureValues(xdata_adjustRT, value = "into"))


ft_ints <- featureValues(xdata_adjustRT, value = "into")

pc <- prcomp(t(na.omit(ft_ints)), center = TRUE)

plot(pc$x[,1], pc$x[,2])

