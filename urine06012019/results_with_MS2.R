ms2p <- readMSData(fnames_pos, mode='onDisk')
ms2n <- readMSData(fnames_neg, mode='onDisk')

#Peak finding
ms2data_p <- findChromPeaks(ms2p, param=cwp, BPPARAM = MulticoreParam(8))
system('echo 123qwe | sudo -S swapoff -a')
Sys.sleep(3)
system('echo 123qwe | sudo -S swapon -a')

ms2data_n <- findChromPeaks(ms2n, param=cwp, BPPARAM = MulticoreParam(8))
system('echo 123qwe | sudo -S swapoff -a')
Sys.sleep(3)
system('echo 123qwe | sudo -S swapon -a')

#Alignment with obiwarp algorithm
ms2data_p$sample_type = "test"
ms2data_n$sample_type = "test"

ms2data_p$sample_type[c(8,13,18,23,28,33,38,43,48,53)] <- 'QC'
ms2data_n$sample_type[c(5,10,15,20,25,30,35,40,45)] <- 'QC'

owp_ms2_p <- ObiwarpParam(subset=which(ms2data_p$sample_type=='QC'),
                           subsetAdjust='average')
owp_ms2_n <- ObiwarpParam(subset=which(ms2data_n$sample_type=='QC'),
                          subsetAdjust='average')
ms2data_p <- adjustRtime(ms2data_p, param=owp_ms2_p)
ms2data_n <- adjustRtime(ms2data_n, param=owp_ms2_n)

#Correspondence
pdpms2p <- PeakDensityParam(sampleGroups = rep(1, length(fileNames(ms2data_p))),
                            minFraction = 0.4, bw = 5)
pdpms2n <- PeakDensityParam(sampleGroups = rep(1, length(fileNames(ms2data_n))),
                            minFraction = 0.4, bw = 5)

ms2data_p <- groupChromPeaks(ms2data_p, param = pdpms2p)
ms2data_n <- groupChromPeaks(ms2data_n, param = pdpms2n)
system('echo 123qwe | sudo -S swapoff -a')
Sys.sleep(3)
system('echo 123qwe | sudo -S swapon -a')

#Fill missing values use less CPUs
ms2data_p <- fillChromPeaks(ms2data_p, BPPARAM = MulticoreParam(2))
ms2data_n <- fillChromPeaks(ms2data_n, BPPARAM = MulticoreParam(2))

write.csv(featureSpectra(ms2data_p, return.type='list'), 'ms2_pos.txt')
system('echo 123qwe | sudo -S swapoff -a')
system('echo 123qwe | sudo -S swapon -a')

write.csv(ms2info_p, '')

save.image()
