library(devtools)
load_all()
load_all("/home/probst/Paper/Exploration_of_Hyperparameters/OMLbots")

lrn.par.set = getMultipleLearners()
simple.lrn.par.set = getSimpleLearners()

# Database extraction

path = paste0("/home/probst/Paper/Exploration_of_Hyperparameters/OMLbots", "/mlrRandomBotDatabaseSnapshot.db")
local.db = src_sqlite(path, create = FALSE)

tbl.results = collect(tbl(local.db, sql("SELECT * FROM [tbl.results]")), n = Inf)
# maybe use less results, as runtime is too high...
tbl.metaFeatures = collect(tbl(local.db, sql("SELECT * FROM [tbl.metaFeatures]")), n = Inf)
tbl.hypPars = collect(tbl(local.db, sql("SELECT * FROM [tbl.hypPars]")), n = Inf)
tbl.runTime = collect(tbl(local.db, sql("SELECT * FROM [tbl.runTime]")), n = Inf)
tbl.scimark = collect(tbl(local.db, sql("SELECT * FROM [tbl.scimark]")), n = Inf)
tbl.resultsReference = collect(tbl(local.db, sql("SELECT * FROM [tbl.resultsReference]")), n = Inf)

# get learner names
library(stringi)
learner.names = paste0("mlr.", names(lrn.par.set))
learner.names = stri_sub(learner.names, 1, -5)
# get task.ids only for datasets with more than 200 experiments for each learner
task.ids = calculateTaskIds(tbl.results, tbl.hypPars, min.experiments = 200)
# set surrogate model
surrogate.mlr.lrn = makeLearner("regr.ranger", par.vals = list(num.trees = 2000, num.threads = 10))

surrogate.measures = surrogate.time = list()

for (i in seq_along(learner.names)) {
  set.seed(123 + i)
  print(i)
  surrogate.measures[[i]] = makeSurrogateModel(measure.name = "area.under.roc.curve", 
    learner.name = learner.names[i], task.ids, lrn.par.set, tbl.results, tbl.hypPars, 
    tbl.metaFeatures, tbl.runTime, tbl.scimark, tbl.resultsReference, surrogate.mlr.lrn)
  
  surrogate.time[[i]] = makeSurrogateModel(measure.name = "area.under.roc.curve", 
    learner.name = learner.names[i], task.ids, lrn.par.set, tbl.results, tbl.hypPars, 
    tbl.metaFeatures, tbl.runTime, tbl.scimark, tbl.resultsReference, surrogate.mlr.lrn, time = TRUE)
  save(surrogate.measures, surrogate.time, file = "surrogates.RData")
}
names(surrogate.measures) = learner.names
names(surrogate.time) = learner.names


# Compare different surrogate models (this takes a lot of time with the new datasets)
surrogate.mlr.lrns = list(
  makeLearner("regr.rpart"),
  makeLearner("regr.ranger", par.vals = list(num.trees = 2000, respect.unordered.factors = TRUE, num.threads = 6)),
  makeLearner("regr.cubist"),
  makeLearner("regr.kknn"),
  makeLearner("regr.lm"),
  makeLearner("regr.featureless")
)

surrogate.measures.benchmark = surrogate.time.benchmark = list()
set.seed(123)
for (i in seq_along(learner.names)) {
  print(i)
  surrogate.measures.benchmark[[i]] = makeSurrogateModel(measure.name = "area.under.roc.curve", 
    learner.name = learner.names[i], task.ids, lrn.par.set, tbl.results, tbl.hypPars, 
    tbl.metaFeatures, tbl.runTime, tbl.scimark, tbl.resultsReference, surrogate.mlr.lrns, benchmark = TRUE)
 
  surrogate.time.benchmark[[i]] = makeSurrogateModel(measure.name = "area.under.roc.curve", 
    learner.name = learner.names[i], task.ids, lrn.par.set, tbl.results, tbl.hypPars, 
    tbl.metaFeatures, tbl.runTime, tbl.scimark, tbl.resultsReference, surrogate.mlr.lrns, benchmark = TRUE, time = TRUE)
}

names(surrogate.measures.benchmark) = learner.names
names(surrogate.time.benchmark) = learner.names
save(surrogate.measures.benchmark, surrogate.time.benchmark, file = "surrogates_benchmark.RData")

for(i in seq_along(learner.names)) {
  print(paste(learner.names[i], "---------------------------------------------------------------------------------------"))
  print(surrogate.measures.benchmark[[i]]$result)
}

for(i in seq_along(learner.names)) {
  print(paste(learner.names[i], "---------------------------------------------------------------------------------------"))
  print(surrogate.time.benchmark[[i]]$result)
}
# cubist and ranger are the best surrogate models for the measures
# Outliers destroy mse result for surrogate models for the time. RSQ: ranger and cubist in general the best model
###################### 

#create pareto-front 
#pick random points from pareto-front for validation runs to check results
library(emoa)
load("surrogates.RData")

meta.features = spread(tbl.metaFeatures, quality, value) %>% select(., -data_id)
meta.features = meta.features[2,] 

pdf("paretofronts_example.pdf", width = 10, height = 8)
for(i in seq_along(learner.names)) {
  print(i)
  print(learner.names[i])
  par.front = createParetoFront(learner.name = learner.names[i], lrn.par.set, surrogates.measures = surrogate.measures, surrogates.time = surrogate.time, meta.features, n.points = 10000) 
  plotParetoFront(learner.names[i], par.front)
}
dev.off()

plotParetoFront(learner.names[i], par.front, plotly = TRUE)
plotParetoFront(learner.names[i], par.front, plotly = FALSE)
