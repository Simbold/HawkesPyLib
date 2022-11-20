<%!
    from pdoc.html_helpers import minify_css
%>
<%def name="homelink()" filter="minify_css">
    .homelink {
        display: block;
        font-size: 2em;
        font-weight: bold;
        color: #555;
        padding-bottom: 0.7em;
        border-bottom: 1px solid silver;
    }
    .homelink:hover {
        color: inherit;
    }
    .homelink img {
        max-width:40%;
        max-height: 20em;
        margin: auto;
        margin-bottom: .2em;
    }
</%def>

<style>${homelink()}</style>
<link rel="canonical" href="https://simbold.github.io/HawkesPyLib/${module.url()[:-len('index.html')] if module.is_package else module.url()}">
<link rel="icon" href="https://simbold.github.io/HawkesPyLib/logo.png">
<meta name="google-site-verification" content="UCGhKLVyQvQFsZiNZcKzrWsKdsdJMHr3427rABF6eGE" />