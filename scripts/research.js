$().ready(function(){
    $(".toggleblurb").hide();
    $(".readmore").click(function(){
      $(this).parent().parent().parent().find('.toggleblurb').slideToggle();
    });
  });

$().ready(function(){
  $(".citation").hide();
  $(".reference").click(function(){
    $(this).next('.citation').slideToggle();
  });
});