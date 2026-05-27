# github pages

## notes

github-pages uses `github-pages` gem which is not compatible with latest `jekyll`
so build is done using custom github actions

## create empty branch for github pages

```shell
git clone https://github.com/vladmandic/sdnext sdnext-pages
cd sdnext-pages
git switch --orphan pages
git commit --allow-empty -m "gh-pages create empty"
git push -u origin pages
git log --oneline
```

## create site

```shell
sudo apt-get install ruby-full build-essential zlib1g-dev
jekyll new --skip-bundle .
bundle install
```

## run locally

```shell
bundle exec jekyll serve
```
