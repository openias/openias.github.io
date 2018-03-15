---
layout: post
title:  "Bayesian Wi&#8209;Fi"
date:   2016-10-22 11:55:52 +0200
categories: wireless
author_name: Bj&ouml;rn Smedman
author_url: /author/bjorn
author_avatar: bjorn
show_avatar: true
read_time: 15
feature_image: feature-wifi-router-circuit-board
---
*TLDR:* Bayes rule is cool. Stable Wi&#8209;Fi rules. The former can give us the latter.

Transmitting Wi&#8209;Fi frames over noisy radio spectrum is far from trivial. You see, every second
or so the Wi&#8209;Fi driver in your router needs to make thousands of nitty-gritty little decisions:
Which modulation rate should the Wi&#8209;Fi chipset use for this particular data frame? Should several
frames be aggregated together and sent in a single radio burst? Should the transmission be protected
with a [RTS-CTS handshake](https://en.wikipedia.org/wiki/IEEE_802.11_RTS/CTS)? How many times should
the Wi&#8209;Fi chipset retry the transmission if no acknowledgement frame is received back? And
if all the transmission attempts fail, then what? Should we try again with different parameters,
or simply drop the frame?

Most of the time the Wi&#8209;Fi driver in your router makes sensible decisions, and you enjoy stable
and fast Wi&#8209;Fi. But, as we've all experienced, there are times when your Wi&#8209;Fi turns
less than stellar. In [my experience](http://www.anyfinetworks.com/company) this is often because
the Wi&#8209;Fi driver in your router has become "overwhelmed" by a challenging radio environment,
one in which the usual "rules of thumb" no longer work very well.

So what can we do to improve the stability and performance of Wi&#8209;Fi? How
about replacing the "rules of thumb" with the one rule that rules them all:
[Bayes rule](https://en.wikipedia.org/wiki/Bayes%27_rule). This is the first post in a series that will
explore this opportunity to advance the frontiers of knowledge and push the human race forward. *:P*
With a bit of luck we will some day arrive at an implementation of a Bayesian Wi&#8209;Fi rate control
algorithm for the Linux kernel, so that anyone can run it on a low-cost consumer Wi&#8209;Fi router
(by replacing its firmware with [OpenWrt](http://www.openwrt.org)).

## The Problem and the Usual Rules of Thumb

The Linux kernel defines an
[interface for rate control algorithms in ```net/mac80211.h```](http://lxr.free-electrons.com/source/include/net/mac80211.h?v=4.8#L5330).
So in very practical terms the problem is to implement the operations needed to fill in a ```struct
rate_control_ops```. The main challenges are ```get_rate``` which, as the name implies, is called
for each frame before it is transferred to the Wi&#8209;Fi chipset for transmission over the air, and
```tx_status``` which is called after transmission to report back the result. The former should pick
the modulation rates and other radio parameters to use for transmission, and pack them into an array of
[```ieee80211_tx_rate```s](http://lxr.free-electrons.com/source/include/net/mac80211.h?v=4.8#L795),
each with its own
[```mac80211_rate_control_flags```](http://lxr.free-electrons.com/source/include/net/mac80211.h?v=4.8#L741).
The latter should evaluate the result of the transmission and learn from it; so that a better
selection can be made the next time ```get_rate``` is called.


![Wi-Fi chipset DMA](/img/wifi-chipset-dma.jpg)
<p class="caption">Sketch of a Wi&#8209;Fi chipset DMA
chain: So-called DMA descriptors, each containing the memory address of the frame data and radio
parameters (R<sub>1</sub> to R<sub>4</sub>), are chained together before being transferred to the
Wi&#8209;Fi chipset and transmitted over the air. Arrows represent "pointers".</p>

The default rate control algorithm in Linux is called
[Minstrel](https://wireless.wiki.kernel.org/en/developers/documentation/mac80211/ratecontrol/minstrel).
Most of its implementation is in
[the file ```net/mac80211/rc80211_minstrel_ht.c```](http://lxr.free-electrons.com/source/net/mac80211/rc80211_minstrel_ht.c?v=4.8).
It's [widely cited in the literature](https://scholar.google.com/scholar?q=minstrel+rate+control) and
often used as a benchmark when evaluating novel algorithms. In most radio environments it performs
reasonably well, but it's based on very hand-wavy statistical reasoning. Essentially it just keeps
an exponentially weighted moving average (EWMA) of the packet error rate (PER) for each combination
of radio parameters supported by the communicating Wi&#8209;Fi chipsets. It then constructs retry
chains of radio parameters (corresponding to R<sub>1</sub> through R<sub>4</sub> in the sketch above)
based on very simple "rules of thumb":

  1. first try the radio parameters with the highest expected throughput, then

  2. try the radio parameters with the second highest expected throughput, then

  3. try the radio parameters with the highest expected success probability, and finally

  4. try using the lowest modulation rate supported by both the communicating Wi&#8209;Fi chipsets.

From time to time Minstrel will try some unexpected radio parameters in an attempt to learn. It's this
behavior that has given it its name:
[like a wandering minstrel the algorithm will wander around the different rates and sing wherever it can](https://sourceforge.net/p/madwifi/svn/HEAD/tree/madwifi/trunk/ath_rate/minstrel/minstrel.txt).

There are a number of problems with these "rules of thumb" and I suspect that they lead to suboptimal
performance in more challenging radio environments. I could go on and on about them,
but I think it suffices to say that Minstrel is not based on very sound statistical reasoning.

## The State of the Art

So what would a rate control algorithm based on sound statistical inference look like? Fortunately
I'm not the first person to ask that question; there's a large body of research papers on the subject
out there.

To me the most promising theoretical approach seems to be a Bayesian one where we cast the rate
adaptation problem in the shape of a
[multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit): different radio
parameters are like different slot machines in a casino, each with an unknown (but potentially very
different) reward distribution, and the task of the rate control algorithm is similar to that of a
gambler in trying to determining which slot machine to play in order to maximize reward in some sense.
There's an
[excellent paper by Richard Combes, Alexandre Proutiere, Donggyu Yun, Jungseul Ok and Yung Yi titled "Optimal Rate Sampling in 802.11 Systems"](https://people.kth.se/~alepro/pdf/infocom2014.pdf)
that takes this approach.

![Multi-Armed Bandit](/img/multi-armed-bandit.png)
<p class="caption">Each choice of radio parameters can be modeled as a slot machine with an
uncertain reward distribution, transforming the rate adaptation problem into a (restless / contextual)
multi-armed bandit problem. Slide from David Silvers
<a href="http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/XX.pdf">lecture on the trade-off
between exploration and exploitation in muli-armed bandit problems</a>.</p>

So we have a research paper and we need an implementation. How is that research? Not so fast,
[Combes et al](https://people.kth.se/~alepro/pdf/infocom2014.pdf) makes one unfortunate simplifying
assumption:

> If a transmission is successful at a high rate, it has to be successful at a lower rate, and
> similarly, if a low-rate transmission fails, then a transmitting at a higher rate would also fail.

This may sound like a safe assumption but it is simply not true in the complex world of
[MIMO](https://en.wikipedia.org/wiki/MIMO), collisions and interference. Just one example is that
higher rates often have a higher success probability than lower rates when errors are primarily
caused by collisions or interference (as is common in busy radio environments), simply because a
packet sent at a higher rate stays in the air for a shorter period of time and is therefore less
likely to be "shot down" by sporadic interference.

So what can we do? Is there a way to salvage
[Combes et al](https://people.kth.se/~alepro/pdf/infocom2014.pdf)s results? Probably, but I'm not sure
I'm the person to do it. On the other hand I feel rather sure that we could do something in the same
general direction. For example [Pedro A. Ortega has done some interesting work on Thompson sampling
and Bayesian control](http://www.adaptiveagents.org/bayesian_control_rule) that I think could be
applied to Wi-Fi rate adaptation. There's also a large body of research on so-called
[contextual bandit problems](https://scholar.google.com/scholar?q=contextual+bandit+problem) and
[restless bandit problems](https://scholar.google.com/scholar?q=restless+bandit+problem) that could be
applicable.

## First Steps

I can see a couple of ways forward. One is very theoretical and includes
[reading up more on the contextual bandit problem](https://scholar.google.com/scholar?q=contextual+bandits+problem).

Another is more practical and includes modeling the rate adaptation problem as a
[Dynamic Bayesian Network](https://en.wikipedia.org/wiki/Dynamic_Bayesian_network) and applying
[Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling). A first prototype using existing
code for approximate inference in DBNs (like [libDAI](https://staff.fnwi.uva.nl/j.m.mooij/libDAI/)
or [Mocapy++](https://sourceforge.net/projects/mocapy)) and a network simulator like
[NS-3](https://www.nsnam.org/) could probably be put together relatively quickly. This could be an
interesting Master's thesis project, for example.

Yet another approach is even more practical and can be described as "writing a better Minstrel",
directly targeting the Linux kernel. The focus would be on simplifying assumptions like working with
[conjugate priors](https://en.wikipedia.org/wiki/Conjugate_prior). This would be of less scientific
interest, but perhaps more useful in the short term.

Which approach to take depends on who wants to join in the fun. :) We need everything from theoreticians
to testers, so please don't be shy!

## Want to Join In?

Want to join in the fun, **and** get your name on a research paper and/or some pretty cool open source
code? Shoot me an email at <bjorn@openias.org> or fill in the form below!

<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSc1sEx335_JdWxLcfXUggpk9rl41QbFx1jQfjfHqAecdO2DYA/viewform?embedded=true" width="100%" height="1100" frameborder="0" marginheight="0" marginwidth="0">Loading...</iframe>


