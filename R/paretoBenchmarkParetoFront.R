#' @title benchmarkParetoFront
#' Predicts the paretofront for an out-of-bag dataset and compares it to the real paretofront of the dataset.
#'
#' @param measure.name 
#' @param learner.name 
#' @param task.ids 
#' @param lrn.par.set 
#' @param tbl.results 
#' @param tbl.hypPars 
#' @param tbl.metaFeatures 
#' @param tbl.runTime 
#' @param tbl.resultsReference 
#' @param surrogate.mlr.lrn mlr.learner used for creating the surrogate function
#' @param min.experiments take experiments with a least min.experiments
#' @param n.points1 Number of points for first paretofront (prediction)
#' @param n.points2 Number of points for the real paretofront of the dataset
#'
#' @export
benchmarkParetoFront = function(measure.name = "area.under.roc.curve", 
  learner.name, task.ids, lrn.par.set, tbl.results, tbl.hypPars, 
  tbl.metaFeatures, tbl.runTime, tbl.resultsReference, surrogate.mlr.lrn, min.experiments = 100, n.points1 = 10000, n.points2 = 1000) {
  
  # Leave one (or several) dataset(s) out
  datasets = sort(unique(tbl.results$task.id))
  
  for(i in seq_along(datasets)) {
    excl = datasets[i]
    tbl.results.i = tbl.results[tbl.results$task.id != excl, ]
    
    # Create surrogate models
    surrogate.measures = list(makeSurrogateModel(measure.name = measure.name, 
      learner.name = learner.name, task.ids, lrn.par.set, tbl.results.i, tbl.hypPars, 
      tbl.metaFeatures, tbl.runTime, tbl.resultsReference, surrogate.mlr.lrn, min.experiments = 100))
    names(surrogate.measures) = learner.name
    surrogate.time = list(makeSurrogateModel(measure.name = measure.name, 
      learner.name = learner.name, task.ids, lrn.par.set, tbl.results.i, tbl.hypPars, 
      tbl.metaFeatures, tbl.runTime, tbl.resultsReference, surrogate.mlr.lrn, min.experiments = 100, 
      time = TRUE))
    names(surrogate.time) = learner.name
    # Create pareto front for the left out dataset(s)
    meta.features = subset(tbl.metaFeatures, task.id == excl) %>% select(., majority.class.size, minority.class.size, number.of.classes,
      number.of.features, number.of.instances, number.of.numeric.features, number.of.symbolic.features)
    
    par.front = createParetoFront(learner.name = learner.name, lrn.par.set, surrogates.measures = surrogate.measures, surrogates.time = surrogate.time, meta.features, n.points = n.points1) 
    # plotParetoFront(par.front, plotly = TRUE)
    
    # Compare this pareto front with
    # the real pareto front (created by a local surrogate model)
    tbl.results.i = tbl.results[tbl.results$task.id == excl, ]
    
    # Create surrogate models for the real pareto front
    surrogate.measures = list(makeSurrogateModel(measure.name = measure.name, 
      learner.name = learner.name, task.ids, lrn.par.set, tbl.results.i, tbl.hypPars, 
      tbl.metaFeatures, tbl.runTime, tbl.resultsReference, surrogate.mlr.lrn, min.experiments = 100))
    names(surrogate.measures) = learner.name
    surrogate.time = list(makeSurrogateModel(measure.name = measure.name, 
      learner.name = learner.name, task.ids, lrn.par.set, tbl.results.i, tbl.hypPars, 
      tbl.metaFeatures, tbl.runTime, tbl.resultsReference, surrogate.mlr.lrn, min.experiments = 100, 
      time = TRUE))
    names(surrogate.time) = learner.name
    
    # the default value
    load("package_defaults.RData")
    convertPackageDefault = function(defaults, learner.name, meta.features) {
      def = defaults[[learner.name]]
      p = meta.features$number.of.features
      if ("mtry" %in% names(def)) {
        def$mtry = floor(sqrt(p))/p
      }
      if ("gamma" %in% names(def)) {
        def$gamma = 1/p
      }
      def
    }
    def = convertPackageDefault(defaults, learner.name, meta.features)

    par.front1 = createParetoFront(learner.name = learner.name, lrn.par.set, surrogates.measures = surrogate.measures, 
      surrogates.time = surrogate.time, meta.features, n.points = n.points2, extra.points = rbind(def, par.front$non.dominated$hyp.pars))
    plotParetoFront(learner.name, par.front1, plotly = TRUE, log = TRUE, col = c("purple", rep("pink", nrow(par.front$non.dominated$hyp.pars))), cex = 1)
    plotParetoFront(learner.name, par.front1, plotly = FALSE, log = TRUE, col = c("purple", rep("pink", nrow(par.front$non.dominated$hyp.pars))), cex = 1)
    
    # (visualization? with moving points...)
    # is it dominated by the pareto front?
    # Is the best predicted performance point better?
  }
}