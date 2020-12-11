library(data.table)
library(ggplot2)
library(ggpubr)
library(ggsci)
library(dplyr)

BN=200
hydra_thin=5

#true values
true_h2<-paste0('normal.h2')
true_h2<-fread(true_h2,header=F)
true_h2<-true_h2[2:3]

#read betLong and cpnLong files
bet<-fread(paste0('long_format_bayesRR_RC/t_M10K_N_5K.betLong'),header=F)
cpn<-fread(paste0('long_format_bayesRR_RC/t_M10K_N_5K.cpnLong'),header=F)

#apply burn-in
bet <- bet[which(V1>=BN),]
cpn <- cpn[which(V1>=BN),]

#combine both
bet$cpn <- cpn$V3

#colnames
colnames(bet)<-c('it','id','bet','cpn')

#add one to id as bet files are zero-based
bet[,id:=id+1]

#read group file
a <- fread('normal.group')
colnames(a)<-c('id')

#read bim file and combine with a
bim <- fread('t_M10K_N_5K.bim',select=2)
colnames(bim) <- 'snp'
bim$id <- seq(1:nrow(bim))
bim$a <- a$id

#label a
bim$a<-factor(bim$a, levels=c(0,1), labels=c('a1','a2'))

#map to bet
bet.mapped <- merge(bet,bim,by='id',all.x=TRUE)

#variance explained by groups 
bet.mapped[,sum.b2.a:=sum(bet^2),by=list(it,a)]

#variance explained by mixtures
bet.mapped[,sum.b2:=sum(bet^2),by=list(it,a,cpn)]

#keep unique rows
bet.mapped=bet.mapped[,list(it,a,cpn,sum.b2,sum.b2.a)]
dat.a <- unique(bet.mapped[,list(it,a,sum.b2.a)])
dat.cpn <- unique(bet.mapped[,list(it,a,cpn,sum.b2)])

#format
dat.cpn$cpn<-as.factor(dat.cpn$cpn)

credible_interval_95 <- function(x) {
  r <- quantile(x, probs=c(0.025, 0.975))
  names(r) <- c("ymin","ymax")
  r
}

#get mean and credible intervals for the variance explained by mixtures
dat.cpn.mean=as.data.table(aggregate(.~a+cpn, dat.cpn[,c('a','cpn','sum.b2')], mean))
ymin = as.data.table(aggregate(.~a+cpn, dat.cpn[,c('a','cpn','sum.b2')], credible_interval_95))$sum.b2.ymin
ymax = as.data.table(aggregate(.~a+cpn, dat.cpn[,c('a','cpn','sum.b2')], credible_interval_95))$sum.b2.ymax
dat.cpn.mean$ymin <- ymin
dat.cpn.mean$ymax <- ymax

#plot 1

text_size=12
box_quantiles_95 <- function(x) {
  r <- quantile(x, probs=c(0.025, 0.25, 0.5, 0.75, 0.975))
  names(r) <- c("ymin", "lower", "middle", "upper", "ymax")
  r
}
pal=pal_d3("category20c")(20)

p1 <- ggplot(dat.a, aes(x=a,y=sum.b2.a,fill=a)) +
  geom_violin(trim=FALSE, alpha=0.4,lwd=0.2) +
  stat_summary(fun.data = box_quantiles_95, geom="boxplot", width=0.1, fill='white', lwd = 0.2) +
  scale_fill_d3("category20c") +
  theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.spacing = unit(0.2,'cm')) +
  theme(strip.background = element_rect(fill="white")) +
  theme(strip.text=element_text(size = text_size, color = "gray10")) +
  xlab('annotations') + ylab('genetic variance')

p1 <- p1 + geom_hline(yintercept=as.numeric(true_h2[1,2]), linetype="dashed", color = pal[1])
p1 <- p1 + geom_hline(yintercept=as.numeric(true_h2[2,2]), linetype="dashed", color = pal[2])

p2 <- ggplot(dat.cpn.mean, aes(x=cpn,y=sum.b2,fill=a)) +
  geom_bar(stat="identity",width = 0.75, alpha=0.4, lwd=0.3, position = position_dodge(.8))  +
  geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.3, position=position_dodge(.8), lwd=0.3, alpha=0.8) +
  facet_grid(~a) +
  scale_fill_d3("category20c") +
  theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.spacing = unit(0.2,'cm')) +
  theme(strip.background = element_rect(fill="white")) +
  theme(strip.text=element_text(size = text_size, color = "gray10")) +
  xlab('mixtures') + ylab('genetic variance') +
  labs(fill = "annot")


p<-ggarrange(
  p1 + theme(legend.position='none'),p2,
  ncol = 2, nrow = 1,
  labels=c('a','b'), font.label = list(size=text_size,face="plain"),
  widths=c(0.5,1),
  align='hv'
)

pdf('plot.pdf', width=8, height=3)
print(p)
dev.off()

