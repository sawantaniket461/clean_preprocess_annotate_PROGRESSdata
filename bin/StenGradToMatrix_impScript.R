suppressMessages(library(dplyr))
suppressMessages(library(tidyr))

args <- commandArgs(trailingOnly = TRUE)  # Get command-line arguments
outputFolder <- args[1]  # First argument
#print (outputFolder)
dat = read.delim(paste0(outputFolder, "/temp/stenosisGradesCombined_seg1-15.tsv"), stringsAsFactors=F)

# seg1
dat1 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_1_sten_grad))
seg1 = spread(dat1, progress_coro_id, sten_seg_1_sten_grad)
write.table(seg1, paste0(outputFolder, "/temp/sten_seg_1_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg2
dat2 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_2_sten_grad))
seg2 = spread(dat2, progress_coro_id, sten_seg_2_sten_grad)
write.table(seg2, paste0(outputFolder, "/temp/sten_seg_2_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg3
dat3 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_3_sten_grad))
seg3 = spread(dat3, progress_coro_id, sten_seg_3_sten_grad)
write.table(seg3, paste0(outputFolder, "/temp/sten_seg_3_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg4
dat4 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_4_sten_grad))
seg4 = spread(dat4, progress_coro_id, sten_seg_4_sten_grad)
write.table(seg4, paste0(outputFolder, "/temp/sten_seg_4_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg5
dat5 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_5_sten_grad))
seg5 = spread(dat5, progress_coro_id, sten_seg_5_sten_grad)
write.table(seg5, paste0(outputFolder, "/temp/sten_seg_5_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg6
dat6 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_6_sten_grad))
seg6 = spread(dat6, progress_coro_id, sten_seg_6_sten_grad)
write.table(seg6, paste0(outputFolder, "/temp/sten_seg_6_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg7
dat7 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_7_sten_grad))
seg7 = spread(dat7, progress_coro_id, sten_seg_7_sten_grad)
write.table(seg7, paste0(outputFolder, "/temp/sten_seg_7_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg8
dat8 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_8_sten_grad))
seg8 = spread(dat8, progress_coro_id, sten_seg_8_sten_grad)
write.table(seg8, paste0(outputFolder, "/temp/sten_seg_8_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg9
dat9 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_9_sten_grad))
seg9 = spread(dat9, progress_coro_id, sten_seg_9_sten_grad)
write.table(seg9, paste0(outputFolder, "/temp/sten_seg_9_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg10
dat10 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_10_sten_grad))
seg10 = spread(dat10, progress_coro_id, sten_seg_10_sten_grad)
write.table(seg10, paste0(outputFolder, "/temp/sten_seg_10_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg11
dat11 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_11_sten_grad))
seg11 = spread(dat11, progress_coro_id, sten_seg_11_sten_grad)
write.table(seg11, paste0(outputFolder, "/temp/sten_seg_11_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg12
dat12 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_12_sten_grad))
seg12 = spread(dat12, progress_coro_id, sten_seg_12_sten_grad)
write.table(seg12, paste0(outputFolder, "/temp/sten_seg_12_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg13
dat13 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_13_sten_grad))
seg13 = spread(dat13, progress_coro_id, sten_seg_13_sten_grad)
write.table(seg13, paste0(outputFolder, "/temp/sten_seg_13_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg14
dat14 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_14_sten_grad))
seg14 = spread(dat14, progress_coro_id, sten_seg_14_sten_grad)
write.table(seg14, paste0(outputFolder, "/temp/sten_seg_14_sten_grad.tsv"), sep="\t", quote=F, row.names=F)

# seg15
dat15 = subset(dat, select=c(progress_patient_id,progress_coro_id,sten_seg_15_sten_grad))
seg15 = spread(dat15, progress_coro_id, sten_seg_15_sten_grad)
write.table(seg15, paste0(outputFolder, "/temp/sten_seg_15_sten_grad.tsv"), sep="\t", quote=F, row.names=F)
