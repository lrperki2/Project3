library(rmarkdown)
render("onlines_news_pop.Rmd", 
       output_file = "data_channel_is_entertainment.html",
       params = list(channel = "data_channel_is_entertainment"))

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "onlines_news_pop", output_file = x[[1]], params = x[[2]])
      }
)