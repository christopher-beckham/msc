pdf("entropy.pdf")

ps = seq(from=0,to=1,by=0.01)
y_ps = ps*(1-ps)

plot(x=ps, y=y_ps, type="l", lwd=1.5, xlab="p", ylab="f(p) = p(1-p)")

dev.off()