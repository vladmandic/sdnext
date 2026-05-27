---
layout: default
title: SD.Next
permalink: /
description: "SD.Next is a powerful, open-source WebUI app for AI image and video generation, built on Stable Diffusion and supporting dozens of advanced models."
---

<div class="homepage-shell">
  <div class="hero-shell">
    {% assign hero_section = site.data.readme_sections | where: "slug", "hero" | first %}
    {% if hero_section %}
      <section class="hero-card readme-section" id="{{ hero_section.slug }}">
        {% include readme/{{ hero_section.slug }}.html %}
      </section>
    {% endif %}
  </div>

  <div class="content-shell">
    {% for section in site.data.readme_sections %}
      {% unless section.slug == "hero" %}
        <section class="readme-section{% if section.slug == 'table-of-contents' %} toc-card{% endif %}" id="{{ section.slug }}">
          {% include readme/{{ section.slug }}.html %}
        </section>
      {% endunless %}
    {% endfor %}
  </div>
</div>
