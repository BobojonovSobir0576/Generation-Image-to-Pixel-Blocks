from rest_framework import serializers
from django.contrib.auth import get_user_model, authenticate

# Получаем модель пользователя
User = get_user_model()


class UserRegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'password', 'email', 'phone', 'first_name', 'last_name',]

    def create(self, validated_data):
        # Создание нового пользователя с обязательными полями
        create_user = User.objects.create_user(**validated_data)
        create_user.set_password(validated_data['password'])
        return create_user

    def update(self, instance, validated_data):
        # Обновление данных пользователя
        instance.username = validated_data.get('username', instance.username)
        instance.email = validated_data.get('email', instance.email)
        instance.phone = validated_data.get('phone', instance.phone)
        instance.first_name = validated_data.get('first_name', instance.first_name)
        instance.last_name = validated_data.get('last_name', instance.last_name)
        instance.save()
        return instance


class UserLoginSerializer(serializers.Serializer):
    phone = serializers.CharField(max_length=15)
    password = serializers.CharField(
        label="Пароль",
        style={'input_type': 'password'},
        trim_whitespace=False
    )

    def validate(self, attrs):
        phone = attrs.get('phone')
        password = attrs.get('password')

        if phone and password:
            user = authenticate(request=self.context.get('request'),
                                phone=phone, password=password)

            if not user:
                msg = 'Невозможно войти с предоставленными учетными данными.'
                raise serializers.ValidationError(msg, code='authorization')
        else:
            msg = 'Необходимо включить "phone" и "password".'
            raise serializers.ValidationError(msg, code='authorization')

        attrs['user'] = user
        return attrs


class UserDetailSerializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = ['id', 'username', 'first_name', 'last_name', 'phone']
