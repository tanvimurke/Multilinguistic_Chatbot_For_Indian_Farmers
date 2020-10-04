package com.example.myapplication;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.location.Address;
import android.location.Geocoder;
import android.location.Location;
import android.location.LocationManager;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.naturallanguage.FirebaseNaturalLanguage;
import com.google.firebase.ml.naturallanguage.languageid.FirebaseLanguageIdentification;
import com.google.firebase.ml.naturallanguage.translate.FirebaseTranslateLanguage;
import com.google.firebase.ml.naturallanguage.translate.FirebaseTranslator;
import com.google.firebase.ml.naturallanguage.translate.FirebaseTranslatorOptions;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;





public class chat extends AppCompatActivity {
    TextView outputText;
    Button translate;
    TextView english;
    TextView lang;
    String sourceText;
    String sourceAnswer;

    TextView sourceanswer;
    TextView targetanswer;
    Button translateanswer;

    TextToSpeech t1;
    ImageButton speakbutton;

    TextView LocText;

    //VAR SOCKET
    private Socket socket;

    private static final int SERVERPORT = 7000;
    private static final String SERVER_IP = "192.168.225.29";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chat);



        outputText = (TextView) findViewById(R.id.txt_output);
        translate = (Button) findViewById(R.id.translateE);
        english = (TextView) findViewById(R.id.translatedText);
        lang = (TextView) findViewById(R.id.sourceLang);

        sourceanswer = (TextView) findViewById(R.id.sourceans);
        targetanswer = (TextView) findViewById(R.id.targetans);
        translateanswer = (Button) findViewById(R.id.translateans);

        speakbutton = (ImageButton) findViewById(R.id.speak);

        LocText = (TextView) findViewById(R.id.locText);


        translate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {


                callAsyncTask();

            }


        });




        //translate button
        //convert english to hindi
        translateanswer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                TranslateMyAnswer();
                //identify the language
                // identifyLanguage();

            }

        });


        //speaker button
        //says the answer in hindi
        //enable google text-to-speech engine option in settings for this to work
        speakbutton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //text to speech
                t1 = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
                    @Override
                    public void onInit(int i) {
                        if (i == TextToSpeech.SUCCESS) {
                            int lang = t1.setLanguage(new Locale("hin"));
                            String toSpeak = targetanswer.getText().toString();
                            Toast.makeText(getApplicationContext(), toSpeak, Toast.LENGTH_SHORT).show();
                            //t1.speak(toSpeak, TextToSpeech.QUEUE_FLUSH, null);
                            int speech = t1.speak(toSpeak, TextToSpeech.QUEUE_FLUSH, null);
                        }
                    }
                });

            }
        });


    }
    private void callAsyncTask(){
        new Thread(new ClientThread()).start();
        new AsyncCaller().execute();

    }


    /*
    THE MAIN SOCKET PROGRAMMING CODE THRU WHICH APP WILL RUN
     */

    //Use AsyncTask to concurrently run threads and get data from sever in background
    private class AsyncCaller extends AsyncTask<String, Void, String> {
        ProgressDialog pdLoading = new ProgressDialog(chat.this);
        String str_question_for_chatbot = "Init", translated_question_from_user = "pqr", str_Location = "xxxyyyzzz";

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            /*
            In onPreExecute we do following tasks:
             --Get current location--

             */
            pdLoading.setMessage("\tLoading...");
            pdLoading.show();
            //get location (city)
            if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {

                requestPermissions(new String[] {Manifest.permission.ACCESS_COARSE_LOCATION}, 1000);
            }
            else{
                LocationManager locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
                Location location = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
                try {
                    String city = hereLocation(location.getLatitude(), location.getLongitude());
                    str_Location = city;
                    LocText.setText(str_Location);
                } catch (Exception e){
                    e.printStackTrace();
                    Toast.makeText(chat.this, "City Not Found!", Toast.LENGTH_SHORT).show();
                }

            }


        }


        @Override
        protected String doInBackground(String... params) {
            /*
                In diInBackground following tasks are executed:
                    1. Get translated question from UI.
                    2. Create the question to be sent to server,, by appending location to
                       translated question.
                    3. Connect to server.
                    4. Send data and receive answer from bot.
             */
            String response = null;
            while(translated_question_from_user.equals("pqr")){
                try{Thread.sleep(10);}catch(InterruptedException ie){ie.printStackTrace();}
               translated_question_from_user = english.getText().toString();
                Log.d("Pre",translated_question_from_user);

            }
            str_question_for_chatbot = translated_question_from_user + '+' + str_Location;
            Log.e("Question",str_question_for_chatbot);
           try {


                PrintWriter out = new PrintWriter(new BufferedWriter(
                        new OutputStreamWriter(socket.getOutputStream())), true);
                out.println(str_question_for_chatbot);
                out.flush();

                InputStream input = socket.getInputStream();
                int lockSeconds = 10*1000;

                long lockThreadCheckpoint = System.currentTimeMillis();
                int availableBytes = input.available();
                while(availableBytes <=0 && (System.currentTimeMillis() < lockThreadCheckpoint + lockSeconds)){
                    try{Thread.sleep(10);}catch(InterruptedException ie){ie.printStackTrace();}
                    availableBytes = input.available();
                }

                byte[] buffer = new byte[availableBytes];
                input.read(buffer, 0, availableBytes);
                response = new String(buffer);

                out.close();
                input.close();
                socket.close();
            } catch (UnknownHostException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }

            return response;
           //return str_question_for_chatbot;
        }

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            /*
            In onPostExecute following tasks are completed:
            Set the result to the answer textbox
             */
            sourceanswer.setText(result);
            pdLoading.dismiss();
        }

    }

    //Connect to Server Socket
    class ClientThread implements Runnable {

        @Override
        public void run() {

            try {
                InetAddress serverAddr = InetAddress.getByName(SERVER_IP);

                socket = new Socket(serverAddr, SERVERPORT);
                if(socket.isConnected()){
                    Log.d("Is Socket Connected?", "Yes");
                }
                else{
                    Log.d("Is Socket Connected?", "NOPE :(((((");
                }

            } catch (UnknownHostException e1) {
                e1.printStackTrace();
            } catch (IOException e1) {
                e1.printStackTrace();
            }

        }

    }


    //location permission
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        //super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case 1000: {
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    LocationManager locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
                    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                        // TODO: Consider calling
                        //    ActivityCompat#requestPermissions
                        // here to request the missing permissions, and then overriding
                        //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                        //                                          int[] grantResults)
                        // to handle the case where the user grants the permission. See the documentation
                        // for ActivityCompat#requestPermissions for more details.
                        return;
                    }
                    Location location = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
                    try {
                        String city = hereLocation(location.getLatitude(), location.getLongitude());
                        LocText.setText(city);
                    } catch (Exception e){
                        e.printStackTrace();
                        Toast.makeText(chat.this, "City Not Found!", Toast.LENGTH_SHORT).show();
                    }
                }
                else{
                    Toast.makeText(this, "Permission not granted!", Toast.LENGTH_SHORT).show();
                }
                break;
            }
        }
    }

    //location
    private String hereLocation(double lat, double lon) {
        String cityName = "";

        Geocoder geocoder = new Geocoder(this, Locale.getDefault());
        List<Address> addresses;
        try{
            addresses = geocoder.getFromLocation(lat, lon, 10);
            if(addresses.size() > 0){
                for (Address adr : addresses){
                    if (adr.getLocality() != null && adr.getLocality().length() > 0){
                        cityName = adr.getLocality();
                        break;
                    }
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return  cityName;
    }



    //gets speech and identifies the language
    private void identifyLanguage() {
        sourceText = outputText.getText().toString();
        FirebaseLanguageIdentification identifier = FirebaseNaturalLanguage.getInstance().getLanguageIdentification();
        lang.setText("Detecting...");
        identifier.identifyLanguage(sourceText).addOnSuccessListener(new OnSuccessListener<String>() {
            @Override
            public void onSuccess(String s) {
                if(s.equals("und")){
                    Toast.makeText(getApplicationContext(), "Language Not Identified", Toast.LENGTH_SHORT).show();

                }
                else{
                    getLanguageCode(s);
                }
            }
        });

    }

    //language codes for indian languages
    private void getLanguageCode(String language) {
        int langCode;
        switch(language){
            case "hi" :
                Log.d("LangCode","Hindi");

                langCode = FirebaseTranslateLanguage.HI;
                lang.setText("Hindi");
                break;

            case "bn" :
                langCode = FirebaseTranslateLanguage.BN;
                lang.setText("Bengali");
                break;


            case "gu" :
                langCode = FirebaseTranslateLanguage.GU;
                lang.setText("Gujarati");
                break;

            case "kn" :
                langCode = FirebaseTranslateLanguage.KN;
                lang.setText("Kannada");
                break;

            case "mr" :
                langCode = FirebaseTranslateLanguage.MR;
                lang.setText("Marathi");
                break;

            case "ta" :
                langCode = FirebaseTranslateLanguage.TA;
                lang.setText("Tamil");
                break;

            case "te" :
                langCode = FirebaseTranslateLanguage.TE;
                lang.setText("Telugu");
                break;

            default:
                langCode= 0;
        }


//after identifying then convert it to english
        translateText(langCode);
        Log.d("LangCode",""+langCode);

    }

    //converts to english language from the language that is identified
    private void translateText(int langCode) {
        english.setText("Translating...");
        FirebaseTranslatorOptions options = new FirebaseTranslatorOptions.Builder().setSourceLanguage(langCode).setTargetLanguage(FirebaseTranslateLanguage.EN).build();

        final FirebaseTranslator translator = FirebaseNaturalLanguage.getInstance().getTranslator(options);

        FirebaseModelDownloadConditions conditions = new FirebaseModelDownloadConditions.Builder().build();

        translator.downloadModelIfNeeded(conditions).addOnSuccessListener(new OnSuccessListener<Void>() {
            @Override
            public void onSuccess(Void aVoid) {

                translator.translate(sourceText).addOnSuccessListener(new OnSuccessListener<String>() {
                    @Override
                    public void onSuccess(String s) {
                        english.setText(s);
                    }
                }).addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {


                    }
                });
            }
        });
    }


   //translate answer from english to hindi
   private void TranslateMyAnswer() {

       sourceAnswer = sourceanswer.getText().toString();

       FirebaseTranslatorOptions options = new FirebaseTranslatorOptions.Builder().setSourceLanguage(FirebaseTranslateLanguage.EN).setTargetLanguage(FirebaseTranslateLanguage.HI).build();

       final FirebaseTranslator translator = FirebaseNaturalLanguage.getInstance().getTranslator(options);

       FirebaseModelDownloadConditions conditions = new FirebaseModelDownloadConditions.Builder().build();

       translator.downloadModelIfNeeded(conditions).addOnSuccessListener(new OnSuccessListener<Void>() {
           @Override
           public void onSuccess(Void aVoid) {

               translator.translate(sourceAnswer).addOnSuccessListener(new OnSuccessListener<String>() {
                   @Override
                   public void onSuccess(String s) {

                       targetanswer.setText(s);

                   }

               }).addOnFailureListener(new OnFailureListener() {
                   @Override
                   public void onFailure(@NonNull Exception e) {

                   }
               });
           }
       });

       //String question = targetanswer.getText().toString() + LocText.getText().toString();

   }



    //this is for mic button- it recognizes our voice
    //use google voice typing for this to work(set lang as primary)
    public void btnSpeech(View view) {

        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        //intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, "hi-IN");
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
        intent.putExtra(RecognizerIntent.EXTRA_PROMPT, "ask your question/अपना सवाल पूछो");

        try{
            startActivityForResult(intent, 1);
        }catch(ActivityNotFoundException e){
            Toast.makeText(this, e.getMessage(), Toast.LENGTH_SHORT).show();
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode){
            case 1:
                if(resultCode==RESULT_OK && null!=data){
                    ArrayList<String> result = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
                    outputText.setText(result.get(0));
                    //Important to call this method before Asynctask or _SERVERCONNECT_ threads are called.
                    identifyLanguage();
                }

                break;
        }


    }
}
