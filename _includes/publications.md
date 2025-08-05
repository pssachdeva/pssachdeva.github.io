<!-- Group publications by category -->
{% assign categories = site.data.publications.main | group_by:"category"%}

{% for category in categories %}
<div class="category-section">
  <div class="category-header"
       onclick="toggleCategory('{{ category.name | replace:' ','-' | downcase }}')">
    <h2 style="margin:0;display:inline">{{ category.name }}</h2>
    <span class="category-toggle" id="toggle-{{ category.name | replace:' ','-' | downcase }}">▼</span>
  </div>

  <div class="category-content" id="content-{{ category.name | replace:' ','-' | downcase }}">
    {% for link in category.items %}
    <div class="publication-entry" style="padding:0 1rem 1rem">
      <h3 style="margin:0">
        {% if link.pdf %}
        <a href="{{ link.pdf }}" style="text-decoration:none;color:inherit">{{ link.title }}</a>
        {% else %}{{ link.title }}{% endif %}
      </h3>
      <p style="margin:0">{{ link.authors }}</p>
      <p class="publication-venue">{{ link.conference }}{% if link.journal %}{{ link.journal }}{% endif %}</p>
      <div>
        {% if link.pdf %}<a href="{{ link.pdf }}" class="pub-button">PDF</a>{% endif %}
        {% if link.code %}<a href="{{ link.code }}" class="pub-button">Code</a>{% endif %}
        {% if link.others %}{{ link.others }}{% endif %}
        {% if link.notes %}<span style="color:#e74d3c;font-weight:bold;font-style:italic">{{ link.notes }}</span>{% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
</div>
{% endfor %}


<!---- script -->
<script>
function toggleCategory(cat){
  const content = document.getElementById('content-'+cat);
  const arrow   = document.getElementById('toggle-'+cat);

  if(content.style.maxHeight && content.style.maxHeight!=='0px'){
    content.style.maxHeight = '0';
    arrow.textContent = '▼';
  }else{
    content.style.maxHeight = content.scrollHeight + 'px';
    arrow.textContent = '▲';
  }
}
window.addEventListener('load',()=>{         // collapse all on first paint
  document.querySelectorAll('.category-content')
          .forEach(c=>c.style.maxHeight='0');
});
</script>