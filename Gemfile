source "https://rubygems.org"

gem "jekyll", "~> 4.4.1"

gem "kramdown-parser-gfm"

group :jekyll_plugins do
  gem "jekyll-seo-tag"
end

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds since newer versions of the gem do not have a Java counterpart.
gem "http_parser.rb", "~> 0.6.0", platforms: [:jruby]
