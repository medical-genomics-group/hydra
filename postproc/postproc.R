#Rscrip to add header to .csv from BayesRR-RC
require(tidyverse)
read_hyper <- function(csv_file){
    hyper_file <- read_csv(
        csv_file,
        col_names=FALSE)
    n_grps <- hyper_file[1, 2][[1]]
    grps <- 1:n_grps
    the_header <- "iter"
    the_header <- c(the_header, "num_groups")
    the_header <- c(the_header, sapply(grps,function(x)paste0("sigmaG_",x)))
    the_header <- c(the_header, "sigmaE")
    the_header <- c(the_header, "h2")
    the_header <- c(the_header, "num_markers")
    the_header <- c(the_header, "num_groups")
    the_header <- c(the_header, "num_mixtures")
    n_mixtures <- hyper_file[1, length(the_header)][[1]]
    mixtures <- 1:n_mixtures
    for(i in 1:n_grps)
        for(j in 1:n_mixtures)
            the_header <- c(the_header,paste(c("pi_",i,",",j), collapse=""))
    colnames(hyper_file) <- the_header
    hyper_file
}
