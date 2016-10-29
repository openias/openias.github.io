--- 
layout: post
title:  "Bayesian Wi&#8209;Fi: Materials &amp; Methods"
date:   2016-10-29 21:19:02 +0200
categories: wireless
author_name: Bj&ouml;rn Smedman
author_url: /author/bjorn
author_avatar: bjorn
show_avatar: true
read_time: 10
feature_image: feature-wifi-materials
---
First of all, a big thanks for the great response to the [previous post in this
series]({% post_url 2016-10-22-bayesian-wifi-rate-control %}). For a brief time it
occupied the top spot in Reddit's [/r/Statistics](http://www.reddit.com/r/statistics) and
[/r/MachineLearning](http://www.reddit.com/r/MachineLearning).

![Google Analytics Screenshot](/img/google-analytics-screenshot.png)
<p class="caption">Just in case anybody's curious, this is what you can expect in terms of traffic from
a top spot in <a href="http://www.reddit.com/r/statistics">/r/Statistics</a> and
<a href="http://www.reddit.com/r/MachineLearning">/r/MachineLearning</a>.</p>

The [last post]({% post_url 2016-10-22-bayesian-wifi-rate-control %}) ended with a speculation on
next steps and a request for collaborators. Since most of the interest has come from academics I'm
leaning towards a paper as the primary deliverable, with an implementation submitted to the Linux
kernel coming in a strong second place.

## Materials & Methods

The other day I went out and bought a
[TP&#8209;Link TL&#8209;WDN4800](http://www.tp-link.com/en/products/details/cat-11_TL-WDN4800.html)
Wi&#8209;Fi card for my desktop and a
[TP&#8209;Link TL&#8209;WA901ND v4](http://www.tp-link.com/en/products/details/cat-12_TL-WA901ND.html)
wireless access point. Luckily it was very straightforward to install
[LEDE](https://www.lede-project.org/) (a fork of [OpenWrt](https://openwrt.org/)) on the access point,
and the Wi&#8209;Fi card just worked instantly with [Ubuntu 16.04](http://releases.ubuntu.com/16.04/).

![Wi-Fi Rates Screenshot](/img/wifi-rc_stats.png)
<p class="caption">Minstrel's rate table contains 52 combinations of radio parameters to consider for
the connection between my newly bought Wi&#8209;Fi card and access point: 4 legacy 802.11b rates, 8
single-stream 802.11n rates with short
<a href="https://en.wikipedia.org/wiki/Guard_interval">guard interval</a>, 8
<a href="https://en.wikipedia.org/wiki/Spatial_multiplexing">dual-stream</a> 802.11n rates with short
guard interval, 8 triple-stream 802.11n rates with short guard interval, and then all the 802.11n
rates again, this time with a long guard interval instead.</p>

I've also set up a [GitHub repository for the project](https://github.com/bjornsing/bayesian-wifi).
So far it contains a
[```Makefile```](https://github.com/bjornsing/bayesian-wifi/blob/d1e05eac11751cc8c530699c0bec404797854d7a/Makefile)
to make it easy to clone the [LEDE](https://www.lede-project.org/) git repository, apply some patches
and build firmware for a number of consumer Wi&#8209;Fi routers, among them my
[TP&#8209;Link TL&#8209;WA901ND v4](http://www.tp-link.com/en/products/details/cat-12_TL-WA901ND.html).
The patches make Minstrel
[sample a lot more](https://github.com/bjornsing/bayesian-wifi/blob/d1e05eac11751cc8c530699c0bec404797854d7a/mac80211-patches/990-mac80211-sample-alot.patch)
and [log the result of transmissions](https://github.com/bjornsing/bayesian-wifi/blob/524cea24574ec18482337a3e2ee299b1e15d20be/mac80211-patches/991-mac80211-print-xmit.patch).
A [very simple script called ```log2csv.py``` takes log prints and turns them into comma-separated
values](https://github.com/bjornsing/bayesian-wifi/blob/524cea24574ec18482337a3e2ee299b1e15d20be/bin/log2csv.py),
fit for consumption in a
[Jupyter notebook](https://github.com/bjornsing/bayesian-wifi/blob/0adf2612503b1372276a0a4304386ec598a79cfb/explore.ipynb).

In order to make it easy to jump in without buying gear I've
[added a couple of datasets to the git repo](https://github.com/bjornsing/bayesian-wifi/commit/f3b92911551808dbeff149b141eaa5db1594180c),
so you can just
```git clone https://github.com/bjornsing/bayesian-wifi.git ; cd bayesian-wifi ; jupyter notebook```
and start hacking away. :)

## Next Steps

So what should be the aim of the explorative data analysis? Well, my first hunch is that there's
probably a significant difference between the probability of a transmission with a particular set of
radio parameters succeeding, and the **conditional** probability of that same transmission succeeding
**given that another transmission has just failed**. Since Minstrel does not exploit this pattern in
the data it could be "low hanging fruit" for a new rate control algorithm.

If you want to explore the above hypothesis, or if you have one of your own, feel free to
```git clone https://github.com/bjornsing/bayesian-wifi.git``` and send me a pull request!

## Want to Join In?

Want to join in the fun, **and** get your name on a research paper? Shoot me an email at
<bjorn@openias.org> or fill in the form below!

<iframe
src="https://docs.google.com/forms/d/e/1FAIpQLSc1sEx335_JdWxLcfXUggpk9rl41QbFx1jQfjfHqAecdO2DYA/viewform?embedded=true"
width="100%" height="1100" frameborder="0" marginheight="0" marginwidth="0">Loading...</iframe>


