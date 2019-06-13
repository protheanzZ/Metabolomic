QCs <- list.files('~/movingdisk/urine_6_1_centroid/', pattern='QC_[0-9]_pos.mzXML')
for (i in c(1:length(QCs))){
    QCs[i] <- paste0('~/movingdisk/urine_6_1_centroid/', QCs[i])
}
QCs
QCs <- readMSData(files=QCs, mode='onDisk', msLevel. = 1)
bpis_QC_pos <- chromatogram(QCs, aggregationFun='max')
colors <- paste0(brewer.pal(9, "Set1"), "60")
colors
plot(bpis_QC_pos, col=colors)

QCs <- list.files('~/movingdisk/urine_6_1_centroid/', pattern='QC_[0-9]_neg.mzXML')
for (i in c(1:length(QCs))){
    QCs[i] <- paste0('~/movingdisk/urine_6_1_centroid/', QCs[i])
}
QCs
QCs <- readMSData(files=QCs, mode='onDisk', msLevel. = 1)
bpis_QC_pos <- chromatogram(QCs, aggregationFun='max')
colors <- paste0(brewer.pal(9, "Set1"), "60")
colors
plot(bpis_QC_pos, col=colors)

QCs <- list.files('~/movingdisk/urine_6_1_centroid/', pattern='Filter_050_[0-9]_neg.mzXML')
for (i in c(1:length(QCs))){
    QCs[i] <- paste0('~/movingdisk/urine_6_1_centroid/', QCs[i])
}
QCs
QCs <- readMSData(files=QCs, mode='onDisk', msLevel. = 1)
bpis_QC_pos <- chromatogram(QCs, aggregationFun='max')
colors <- paste0(brewer.pal(3, "Set3"), "90")
plot(bpis_QC_pos, col=colors)

rtr <- c(40, 100)
chr_raw <- chromatogram(QCs, rt = rtr)
plot(chr_raw)

rtr <- c(340, 370)
chr_raw <- chromatogram(QCs, rt = rtr)
plot(chr_raw)
