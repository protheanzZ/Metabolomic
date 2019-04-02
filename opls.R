library(ropls)

data <- read.csv('OPLS.csv', header=T, encoding = 'utf-8',
                 check.names=F)
data['sample name'] <- data[1]

pls <- opls(data[2:(length(data)-2)], data$category, predI=1, orthoI=NA)
pls@vipVn
