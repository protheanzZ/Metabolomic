library(xcms)
library(RColorBrewer)
library(faahKO)
library(magrittr)
library(pander)
library(CAMERA)


get_fnames <- function(path, pattern){
    fnames <- list.files(path, pattern=pattern)
    for (i in c(1:length(fnames))){
        fnames[i] <- paste0(path, fnames[i])
    }
    fnames
}
#Get filepaths using get_fnames predefined
condQCs <- get_fnames(path='~/movingdisk/urine_6_1_centroid/', pattern='.*_cond.*.mzXML')
fnames_pos <- get_fnames(path='~/movingdisk/urine_6_1_centroid/', pattern='.*_pos.mzXML')
fnames_pos <- c(condQCs, fnames_pos[1:45])
fnames_neg <- get_fnames(path='~/movingdisk/urine_6_1_centroid/', pattern='.*_neg.mzXML')

#Read files in 'ondisk' mode
print('Start readData')
print(Sys.time())
raw_data_pos <- readMSData(files=fnames_pos, mode='onDisk', msLevel. = 1)
raw_data_neg <- readMSData(files=fnames_neg, mode='onDisk', msLevel. = 1)
print(Sys.time())
print('Done...')

#Centware peak picking and Switch swap to release memory in harddisk
print('start centwave on positive mode')
print(Sys.time())
cwp <- CentWaveParam(peakwidth = c(5,20), noise = 5000, ppm=2.5)
xdata_pos <- findChromPeaks(raw_data_pos, param = cwp, BPPARAM = MulticoreParam(12))
system('echo 123qwe | sudo -S swapoff -a')
Sys.sleep(3)
system('echo 123qwe | sudo -S swapon -a')

print(Sys.time())
print('start centwave on negative mode')
xdata_neg <- findChromPeaks(raw_data_neg, param = cwp, BPPARAM = MulticoreParam(10))
system('echo 123qwe | sudo -S swapoff -a')
Sys.sleep(3)
system('echo 123qwe | sudo -S swapon -a')
print(Sys.time())
print('Done...')


##Alignment and correspondence

xdata_pos$sample_type <- "test"
xdata_neg$sample_type <- "test"

##Be careful about the indices of QCs
xdata_pos$sample_type[c(8,13,18,23,28,33,38,43,48,53)] <- 'QC'
xdata_neg$sample_type[c(5,10,15,20,25,30,35,40,45)] <- 'QC'
print('adjust begin...')
print(Sys.time())
print('Done...')
owp_pos <- ObiwarpParam(subset=which(xdata_pos$sample_type=='QC'),
                        subsetAdjust='average')
xdata_pos <- adjustRtime(xdata_pos, param=owp_pos)
system('echo 123qwe | sudo -S swapoff -a')
Sys.sleep(3)
system('echo 123qwe | sudo -S swapon -a')

owp_neg <- ObiwarpParam(subset=which(xdata_neg$sample_type=='QC'),
                        subsetAdjust='average')
xdata_neg <- adjustRtime(xdata_neg, param=owp_neg)
system('echo 123qwe | sudo -S swapoff -a')
Sys.sleep(3)
system('echo 123qwe | sudo -S swapon -a')

print(Sys.time())
print('Done...')

print('Correspondece here...')
print(Sys.time())
pdp_pos <- PeakDensityParam(sampleGroups = rep(1, length(fileNames(xdata_pos))),
                        minFraction = 0.4, bw = 5)
xdata_pos <- groupChromPeaks(xdata_pos, param = pdp_pos)
system('echo 123qwe | sudo -S swapoff -a')
Sys.sleep(3)
system('echo 123qwe | sudo -S swapon -a')

pdp_neg <- PeakDensityParam(sampleGroups = rep(1, length(fileNames(xdata_neg))),
                        minFraction = 0.4, bw=5)
xdata_neg <- groupChromPeaks(xdata_neg, param = pdp_neg)
xdata_neg
system('echo 123qwe | sudo -S swapoff -a')
system('echo 123qwe | sudo -S swapon -a')

xdata_pos <- fillChromPeaks(xdata_pos, BPPARAM = MulticoreParam(2))
system('echo 123qwe | sudo -S swapoff -a')
system('echo 123qwe | sudo -S swapon -a')
xdata_neg <- fillChromPeaks(xdata_neg, BPPARAM = MulticoreParam(2))
system('echo 123qwe | sudo -S swapoff -a')
system('echo 123qwe | sudo -S swapon -a')

write.csv(featureDefinitions(xdata_pos)[,1:8], 'pos_definitions2.5.csv')
write.csv(featureDefinitions(xdata_neg)[,1:8], 'neg_definitions2.5.csv')
write.csv(featureValues(xdata_pos, value='into'), 'pos_featureValues2.5.csv')
write.csv(featureValues(xdata_neg, value='into'), 'neg_featureValues2.5.csv')

#CAMERA
xsg_pos <- as(xdata_pos, 'xcmsSet')

xsg_pos <- xsAnnotate(xsg_pos)
#Group after RT value of the xcms grouped peak
xsg_pos <- groupFWHM(xsg_pos, perfwhm=0.6)
#Verify grouping
xsg_pos <- groupCorr(xsg_pos)
#Annotate isotopes, could be done before groupCorr
xsg_pos <- findIsotopes(xsg_pos)
#Annotate adducts
xsg_pos <- findAdducts(xsg_pos, polarity="positive")
#Get final peaktable and store on harddrive
#write.csv(getPeaklist(xsg_pos),file="result_CAMERA_pos.csv")

xsg_neg <- as(xdata_neg, 'xcmsSet')

xsg_neg <- xsAnnotate(xsg_neg)
#Group after RT value of the xcms grouped peak
xsg_neg <- groupFWHM(xsg_neg, perfwhm=0.6)
#Verify grouping
xsg_neg <- groupCorr(xsg_neg)
#Annotate isotopes, could be done before groupCorr
xsg_neg <- findIsotopes(xsg_neg)
#Annotate adducts
xsg_neg <- findAdducts(xsg_neg, polarity="negative")
#Get final peaktable and store on harddrive
#write.csv(getPeaklist(xsg_neg),file="result_CAMERA_neg.csv")

print('Done~!')
print(Sys.time())
save.image('06052019.RData')
