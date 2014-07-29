require(ggplot2)
require(plyr)

save.image <- function(filename) {
  ggsave(paste0('slides/', filename), width=8, height=5.3, units='in')
}

BACKGROUND.COLOR <- '#fdf6e3'
BASELINE.COLOR <- '#c77cff'
EXPECTATION.COLOR <- '#7cad00'

theme_update(
  plot.background=element_rect(color=BACKGROUND.COLOR, fill=BACKGROUND.COLOR),
  legend.background=element_rect(color='transparent', fill='transparent')
)
theme.no.strip.labels <- theme(strip.text=element_blank(), strip.background=element_blank())

MEDIAN.ODDS.RATIO <- 0.85
ODDS.RATIO.SPREAD <- 1.2

rate.to.log.odds <- function(rate) log(rate / (1 - rate))
log.odds.to.rate <- function(log.odds) 1 / (1 + exp(-log.odds))

# Show treatment rate selection

plot.treatment.rate.distribution <- function(baseline.rate) {
  baseline.log.odds <- rate.to.log.odds(baseline.rate)
  treatment.rates <- seq(
    log.odds.to.rate(baseline.log.odds - 1),
    log.odds.to.rate(baseline.log.odds + 1),
    0.001
  )
  log.odds.ratios <- rate.to.log.odds(treatment.rates) - rate.to.log.odds(baseline.rate)
  density <- dnorm(log.odds.ratios, mean=log(MEDIAN.ODDS.RATIO), sd=log(ODDS.RATIO.SPREAD))
  return(
    qplot(treatment.rates, density, geom='line', xlab='Treatment rate')
    + geom_vline(xintercept=baseline.rate, linetype='dashed', col=BASELINE.COLOR)
    + ylab('Density')
    + scale_y_continuous(breaks=NULL)
  )
}

plot.treatment.rate.distribution(0.1)
save.image('treatment_rate_around_0.1.svg')
plot.treatment.rate.distribution(0.5)
save.image('treatment_rate_around_0.5.svg')

# What proportion of treatments are better than the baseline?
pnorm(0, mean=log(MEDIAN.ODDS.RATIO), sd=log(ODDS.RATIO.SPREAD), lower.tail=FALSE)

# Plot results for all decision types

data <- read.csv('newer_final_results.csv')

unique.decision.types.data <- data[!duplicated(data$decision.type),]
ordered.decision.types <- with(
  unique.decision.types.data,
  decision.type[order(test.name, first.parameter, second.parameter)]
)
data <- within(data, {
  decision.type <- factor(decision.type, levels=as.character(ordered.decision.types))
})

summaries <- ddply(
  data,
  .(decision.type, test.name, first.parameter, second.parameter),
  function(sd) {
    with(sd, data.frame(
      mean.final.rate=mean(final.rate, na.rm=TRUE),
      se.of.mean.final.rate=sd(final.rate, na.rm=TRUE) / sqrt(nrow(sd)),
      median.final.rate=median(final.rate, na.rm=TRUE),
      # move single Bayesian parameter to second.parameter for faceting purposes
      second.parameter=if (test.name[1] == 'Bayesian') first.parameter[1] else second.parameter[1],
      first.parameter=if (test.name[1] == 'Bayesian') NA else first.parameter[1]
    ))
  }
)

(ggplot(summaries, aes(mean.final.rate, decision.type))
 + geom_path(aes(group=first.parameter), width=0.5)
 + geom_errorbarh(
     aes(
       xmin=mean.final.rate - 1.96 * se.of.mean.final.rate,
       xmax=mean.final.rate + 1.96 * se.of.mean.final.rate
     ),
     height=0.5,
     color=EXPECTATION.COLOR
   )
 + geom_point(color=EXPECTATION.COLOR)
 + facet_grid(test.name + first.parameter ~ ., scales='free_y', space='free_y')
 + geom_vline(xintercept=0.1, linetype='dashed', color=BASELINE.COLOR)
 + ylab(NULL)
 + xlab('Expected final rate')
 + theme(axis.text.y=element_text(size=8))
 + theme.no.strip.labels
)

save.image('all_types_dotplot.svg')

base.histogram <- ggplot(data) + facet_wrap(~ decision.type, scales='free_y', ncol=4)

(base.histogram + aes(final.rate)
 + geom_bar(binwidth=0.05)
 + geom_vline(xintercept=0.1, linetype='dashed', color=BASELINE.COLOR)
 + geom_vline(aes(xintercept=mean.final.rate), summaries, linetype='dashed',
              color=EXPECTATION.COLOR)
 + xlab('Final rate')
 + theme(text=element_text(size=9), strip.text=element_text(size=6))
 + scale_y_continuous(breaks=NULL) + ylab('Count')
 + scale_x_continuous(limits=c(0, 1))
)

save.image('final_rate_all_types.svg')

# Plot detailed results focused on four decision types

FOCUS.DECISION.TYPES <- c(
  "Bayesian, 0.20% minimum relative lift",
  "Bayesian, 1.00% minimum relative lift",
  "Chisq, 25% significance, 10.0% relative lift",
  "Chisq, 90% significance, 10.0% relative lift"
)

focused.data <- subset(data, decision.type %in% FOCUS.DECISION.TYPES)

focused.base.histogram <- (
  base.histogram %+% focused.data + facet_wrap(~ decision.type, ncol=1, scales="free_y")
  + aes(fill=decision.type) + guides(fill=FALSE) + ylab('Count') + scale_y_continuous(breaks=NULL)
)

focused.base.histogram + aes(final.rate) + geom_bar(binwidth=0.05) + xlab('Final rate')
save.image('final_rate_focused.svg')
focused.base.histogram + aes(total.experiments.run) + geom_bar() + xlab('Number of experiments run')
save.image('total_num_experiments_focused.svg')
focused.base.histogram + aes(loss.from.errors) + geom_bar() + xlab('Total loss from errors')
save.image('total_loss_focused.svg')

# Plot simulation paths for four decision types

path.data <- read.csv('newer_final_paths.csv')
path.data <- within(path.data, run.id <- paste(seed, decision.type))

subset.path.data <- subset(path.data, decision.type %in% FOCUS.DECISION.TYPES)

path.base.plot <- (
  ggplot(subset.path.data, aes(x=num.visitors.seen, y=rate, color=decision.type))
  + xlab('Number of visitors seen')
  + ylab('Conversion rate')
  + theme(legend.position='top', legend.title=element_blank())
  + guides(color=guide_legend(nrow=2))
)

path.base.plot %+% subset(subset.path.data, seed == 0) + geom_line() + geom_point(size=1)
save.image('one_simulation_paths.svg')

(path.base.plot %+% subset(subset.path.data, seed < 9)
 + geom_line()
 + facet_wrap(~ seed)
 + theme.no.strip.labels
)
save.image('nine_simulation_paths.svg')

(path.base.plot + aes(group=run.id)
 + geom_line(alpha=0.1, size=1)
 + facet_wrap(~ decision.type)
 + guides(color=FALSE)
)
save.image('path_clouds.svg')
