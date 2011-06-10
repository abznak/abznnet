refn = function() {
	source("fn.r");
}

load_data = function() {
	dat = read.csv('c:/tims/tmp/ntest2.csv', header=T);
	dat$generation = as.numeric(dat$generation);
	dat$time = as.numeric(dat$time);
	dat$sample = as.numeric(dat$sample);
	dat$got = as.numeric(dat$got);
	dat$want = as.numeric(dat$want);
	dat
}

plot_subset = function(dat) {
	#p = ggplot(data = subset(dat, time = dat[1,]$time)) ;
	p = ggplot(data = dat) + geom_point(aes(x = sample, y = want, color = generation), size = 0) + geom_point(aes(x = sample, y = got), color = "black")
	p
}

plot_ut_text = function() {
	p = plot_ut_abstract() + geom_text(aes(label = p1remaining, colour = p1remaining), size = 3)
	p
}
plot_ut_point = function () {
	p = plot_ut_abstract() + geom_point(aes(size = p1remaining, colour = p1remaining))
	p
}
plot_ut_abstract = function() {
	ut$type1plus = paste("P1 ", ut$type1)
	ut$type2plus = paste("P2 ", ut$type2)
	ut$p1remaining = ut$remain1 - ut$remain2
	ut$p1lost = ut$count1 - ut$remain1

	p = ggplot(data = ut[T | ut$remain1 > ut$remain2,]) + aes(x = count1, y = count2, label = remain1 - remain2)  + facet_grid(type2plus ~ type1plus)
	p = p + geom_abline(color = "#CCCCCC")
	p = p + scale_color_gradient2(low = "red", mid = "black", high = "#00FF00")  + opts(title = "SC2: Player 1 units remaining after an attack-move fight with no micro.")
	p = p + theme_bw()
	p = p + scale_x_continuous('P1 Starting Units') + scale_y_continuous('P2 Starting Units', colour = "green")
	p = p + opts('axis.text.x' = theme_text(colour = "blue"), 'axis.text.y' = theme_text(colour = "blue"))
	p
}
watermark = function() {
	grid.text("http://abznak.com          sc2 v1.0.3", x=.2, y=.01, rot=0, gp = gpar(fontsize=8, col="grey"))
}
# refn(); p = plot_ut(); print(p, vp = viewport(height = .8)); grid.text(0.5, .5, "hello")
