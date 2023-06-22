
//define 
int x=8;
int led_open_eye=9;
int led_close_eye=10;
char y;

// set up out put
void setup()
{
  Serial.begin(9600);  
  pinMode(x,OUTPUT);
  pinMode(led_open_eye,OUTPUT);
  pinMode(led_close_eye,OUTPUT);
}


void open_warning()
{
  digitalWrite(x,HIGH);
  digitalWrite(led_close_eye,HIGH);
  digitalWrite(led_open_eye,LOW);
}
void close_warning()
{
  digitalWrite(x,LOW);
  digitalWrite(led_close_eye,LOW);
  digitalWrite(led_open_eye,HIGH);
}

// main loop 
void loop()
{
      if(Serial.available()>0)
  {
    y=Serial.read();
    Serial.print(x);

    if(y=='1')
    {
      open_warning();
    }
    else if(y=='0')
    {
      close_warning();
    }   
  }
}
